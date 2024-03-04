/*
    This file is part of the implementation of the SIGGRAPH 2020 paper
    "Robust Fitting of Parallax-Aware Mixtures for Path Guiding".
    The implementation extends Mitsuba, a physically based rendering system.

    Copyright (c) 2020 Lukas Ruppert, Sebastian Herholz.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/tls.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/renderproc.h>

#include <mitsuba/guiding/guiding.h>
#include <mitsuba/guiding/guiding_types.h>
#include <mitsuba/guiding/pathguidingintersectiondata.h>
#include <mitsuba/guiding/pathguidingmixturestats.h>
#include <mitsuba/guiding/pathguidingmixturestatsfactory.h>
#include <mitsuba/guiding/IterableBlockedVector.h>
#include <mitsuba/guiding/BSDFOracle.h>
#include <mitsuba/guiding/GuidingField.h>
#include <mitsuba/guiding/GuidedBSDF.h>

#include <atomic>
#include <vector>

MTS_NAMESPACE_BEGIN

using namespace Guiding;

template<>
MTS_EXPORT decltype(BSDFOracle::m_bsdfRepresentaions) BSDFOracle::m_bsdfRepresentaions = {};

static StatsCounter avgPathLength("PathGuiding", "Average path length", EAverage);

#ifdef GUIDING_DETAILED_STATISTICS
static StatsCounter zeroSamples("PathGuiding", "Samples without contribution", EPercentage);
static StatsCounter zeroPaths("PathGuiding", "Paths without contribution", EPercentage);
static StatsCounter zeroPathLengths("PathGuiding", "Average length of paths without contribution", EAverage);
static StatsCounter guidedDirectionSamples("PathGuiding", "Guided direction samples", EPercentage);
static StatsCounter pathsGuidedOutOfScene("PathGuiding", "Paths leaving the scene via guiding", EPercentage);
static StatsCounter pathsNeverIntersectingTheScene("PathGuiding", "Paths never intersecting the scene", EPercentage);

static StatsCounter pathsTerminatedByMaxDepth("Path termination", "maximum depth", EPercentage);
static StatsCounter pathsTerminatedByZeroReflectance("Path termination", "non-reflective material", EPercentage);
static StatsCounter pathsTerminatedByInvalidBSDFDirectionSample("Path termination", "invalid/zero-throughput BSDF sample", EPercentage);
static StatsCounter pathsTerminatedByInvalidGuidedDirectionSample("Path termination", "invalid/zero-throughput guiding sample", EPercentage);
static StatsCounter pathsTerminatedByRussianRoulette("Path termination", "Russian roulette", EPercentage);
static StatsCounter pathsTerminatedByNoIntersection("Path termination", "no intersection", EPercentage);
#endif


template <typename T>
class ScopedValueRestorer
{
private:
    const T oldValue;
    T& currentValue;
public:
    ScopedValueRestorer(T& value) : oldValue{value}, currentValue{value} {}
    ~ScopedValueRestorer() {currentValue = oldValue;}
    const T& operator()() const {return oldValue;}
};

class PathGuiding final : public MonteCarloIntegrator {
private:
    // mutable, since Li(), which is const, has to modify the state of these objects when collecting samples
    mutable std::vector<std::vector<PathGuidingIntersectionData>> m_perThreadIntersectionData;
    mutable std::vector<IterableBlockedVector<PathGuidingSampleData>> m_perThreadSamples;
    mutable std::vector<IterableBlockedVector<Point>> m_perThreadZeroValuedSamples;
    mutable std::atomic<size_t> m_maxThreadId;
    mutable PrimitiveThreadLocal<bool> m_threadLocalInitialized;

    Properties m_guidingFieldProps;
    GuidingFieldType m_guidingField;

    ref<Timer> m_timer;

    AABB m_sceneSpace;
    Float m_envmapDistance;

    std::string m_trainingDirectory;

    std::string m_importTreeFile;
    uint32_t m_minSamplesToStartFitting;
    uint32_t m_trainingSamples;
    uint32_t m_minSPPToTrustGuidingCaches;
    uint32_t m_trainingMaxSeconds;
    uint32_t m_renderMaxSeconds;
    uint32_t m_samplesPerIteration;
    uint32_t m_postponedSPPOfData;
    float m_bsdfProbability;
    bool m_exportIntermediateRenders;
    bool m_exportIntermediateTrees;
    bool m_exportFinalTree;
    bool m_exportIntermediateSamples;
    bool m_renderMeasurementEstimate;
    bool m_renderNumComponents;
    bool m_renderIndirectLight;
    bool m_adrr;
    bool m_useNee;
    bool m_guideDirectLight;
    bool m_accountForDirectLightMiWeight;
    bool m_update;
    bool m_splatSamples;
    bool m_deterministic;
    bool m_usePowerHeuristic;
    bool m_useCosineProduct;
    bool m_useBSDFProduct;
    bool m_disableParallaxAfterTraining;
    bool m_expGrowSamplesPerIteration;

    bool m_collectZeroValuedSamples;
    bool m_training;
    volatile bool m_canceled;

public:

    PathGuiding(const Properties &props)
        : MonteCarloIntegrator(props)
    {
        //render settings
        m_useCosineProduct              = props.getBoolean("useCosineProduct", true);
        m_useBSDFProduct                = props.getBoolean("useBSDFProduct", false);
        m_bsdfProbability               = props.getFloat("bsdfProbability", 0.5f);
        m_adrr                          = props.getBoolean("adrr", false);
        m_useNee                        = props.getBoolean("useNee", true);
        m_guideDirectLight              = props.getBoolean("guideDirectLight", true);
        m_accountForDirectLightMiWeight = props.getBoolean("accountForDirectLightMiWeight", true);
        m_usePowerHeuristic             = props.getBoolean("usePowerHeuristic", false);

        //training/sample generation
        m_minSPPToTrustGuidingCaches    = static_cast<uint32_t>(props.getSize("minSPPToTrustGuidingCaches", 12));
        m_minSamplesToStartFitting      = static_cast<uint32_t>(props.getSize("minSamplesToStartFitting", 128));
        m_deterministic                 = props.getBoolean("deterministic", false);
        m_update                        = props.getBoolean("update", true);
        m_splatSamples                  = props.getBoolean("splatSamples", true);
        m_envmapDistance                = std::numeric_limits<float>::infinity();

        //training spp/time
        m_trainingSamples               = static_cast<uint32_t>(props.getSize("trainingSamples", 32));
        m_samplesPerIteration           = static_cast<uint32_t>(props.getSize("samplesPerIteration", 4));
        m_expGrowSamplesPerIteration    = props.getBoolean("expGrowSamplesPerIteration", false);
        m_trainingMaxSeconds            = static_cast<uint32_t>(props.getSize("trainingMaxSeconds", 0UL));
        //progressive rendering
        m_renderMaxSeconds              = static_cast<uint32_t>(props.getSize("renderMaxSeconds", 0UL));

        if (m_trainingSamples && m_trainingMaxSeconds)
            Log(EError, "training samples and training seconds cannot both be set. please set either one to 0.");

        if ((m_trainingMaxSeconds > 0) != (m_renderMaxSeconds > 0) && (m_trainingSamples || m_trainingMaxSeconds))
            Log(EWarn, "%s is set for limited time but %s is set for limited SPP.", m_trainingMaxSeconds?"training":"rendering", m_renderMaxSeconds?"training":"rendering");

        //debug/visualization render options
        m_renderMeasurementEstimate     = props.getBoolean("renderMeasurementEstimate", false);
        m_renderNumComponents           = props.getBoolean("renderNumComponents", false);
        m_renderIndirectLight           = props.getBoolean("renderIndirectLight", false);
        m_disableParallaxAfterTraining  = props.getBoolean("disableParallaxAfterTraining", false);

        //import/export
        m_exportIntermediateRenders     = props.getBoolean("exportIntermediateRenders", false);
        m_exportIntermediateTrees       = props.getBoolean("exportIntermediateTrees", false);
        m_exportFinalTree               = props.getBoolean("exportFinalTree", false);
        m_exportIntermediateSamples     = props.getBoolean("exportIntermediateSamples", false);
        m_importTreeFile                = props.getString("importTreeFile", "");
        const std::string outputFolder  = props.getString("outputFolder", ".");
        m_trainingDirectory             = props.getString("trainingDirectory", outputFolder+"/training/");

        //fitting/split and merge/parallax configuration
        m_guidingFieldProps.setSize("numInitialComponents", props.getSize("numInitialComponents", 16));
        m_guidingFieldProps.setBoolean("parallaxCompensation", props.getBoolean("parallaxCompensation", true));
        m_guidingFieldProps.setBoolean("safetyMerge", props.getBoolean("safetyMerge", true));
        m_guidingFieldProps.setBoolean("splitAndMerge", props.getBoolean("splitAndMerge", true));
        m_guidingFieldProps.setFloat("splitMinDivergence", props.getFloat("splitMinDivergence", 0.5f));
        m_guidingFieldProps.setFloat("mergeMaxDivergence", props.getFloat("mergeMaxDivergence", 0.025f));
        m_guidingFieldProps.setSize("minSamplesForMerging", props.getSize("minSamplesForMerging", props.getSize("maxSamplesPerLeafNode", 32768)/4));
        m_guidingFieldProps.setSize("minSamplesForSplitting", props.getSize("minSamplesForSplitting", props.getSize("maxSamplesPerLeafNode", 32768)/8));
        m_guidingFieldProps.setSize("minSamplesForPostSplitFitting", props.getSize("minSamplesForPostSplitFitting", props.getSize("maxSamplesPerLeafNode", 32768)/8));
        m_guidingFieldProps.setFloat("decayOnSpatialSplit", props.getFloat("decayOnSpatialSplit", 0.25f));

        //vmm initialization
        m_guidingFieldProps.setSize( "vmmFactory.minItr",                    props.getSize( "emFit.minItr", 1));
        m_guidingFieldProps.setSize( "vmmFactory.maxItr",                    props.getSize( "emFit.maxItr", 100));
        m_guidingFieldProps.setFloat("vmmFactory.relLogLikelihoodThreshold", props.getFloat("emFit.relLogLikelihoodThreshold", 0.005f));
        m_guidingFieldProps.setSize( "vmmFactory.numInitialComponents",      props.getSize( "numInitialComponents", 16));
        m_guidingFieldProps.setFloat("vmmFactory.initKappa",                 props.getFloat("emFit.initKappa", 5.0f));
        m_guidingFieldProps.setFloat("vmmFactory.maxKappa",                  props.getFloat("emFit.maxKappa", 32768.0f));
        m_guidingFieldProps.setFloat("vmmFactory.vPrior",                    props.getFloat("emFit.vPrior", 0.01f));
        m_guidingFieldProps.setFloat("vmmFactory.rPrior",                    props.getFloat("emFit.rPrior", 0.0f));
        m_guidingFieldProps.setFloat("vmmFactory.rPriorWeight",              props.getFloat("emFit.rPriorWeight", 0.2f));

        //kd-tree configuration
        m_guidingFieldProps.setInteger("tree.minSamples", 100);
        m_guidingFieldProps.setSize("tree.maxSamples", props.getSize("maxSamplesPerLeafNode", 32768));
        m_guidingFieldProps.setInteger("tree.maxDepth", props.getInteger("maxTreeDepth", 32));
        m_guidingFieldProps.setInteger("tree.splitType", GuidingFieldType::GuidingTreeType::SplitType::kSampleMeanAndVariance);

        if ((m_exportIntermediateRenders || m_exportIntermediateSamples || m_exportIntermediateTrees || m_exportFinalTree || !m_importTreeFile.empty()) && !fs::exists(m_trainingDirectory))
            fs::create_directory(m_trainingDirectory);

        m_timer = new Timer{false};

        m_collectZeroValuedSamples = false;
        m_training = false;
        m_canceled = false;
    }

    PathGuiding(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager)
    {
        Log(EError, "serialization of the path guiding integrator is not supported.");
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);

        Log(EError, "serialization of the path guiding integrator is not supported.");
    }

    void cancel() {
        m_canceled = true;
        MonteCarloIntegrator::cancel();
    }

    void train(Scene *scene, RenderQueue *queue, const RenderJob *job, int sensorResID) {
        ref<Timer> trainingTimer = new Timer{};

        const ScopedValueRestorer<bool> training{m_training},
                                        collectZeroValuedSamples{m_collectZeroValuedSamples},
                                        adrr{m_adrr}, renderMeasurementEstimate{m_renderMeasurementEstimate},
                                        renderNumComponents{m_renderNumComponents},
                                        accountForDirectLightMiWeight{m_accountForDirectLightMiWeight};
        const ScopedValueRestorer<int> rrDepth(m_rrDepth);

        m_renderMeasurementEstimate = false;
        m_renderNumComponents = false;
        m_collectZeroValuedSamples = m_adrr;

        m_training = true;

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        ref<Scene> trainingScene = new Scene(scene);
        const int trainingSceneResID = sched->registerResource(trainingScene);

        Properties trainingSamplerProps = scene->getSampler()->getProperties();
        if (m_deterministic)
        {
            Log(EInfo, "forcing the use of the deterministic sampler for training");
            trainingSamplerProps.setPluginName("deterministic");
        }

        Log(EInfo, "Starting training with %uspp per iteration%s.", m_samplesPerIteration, (m_expGrowSamplesPerIteration ? " (growing exponentially)" : ""));

        uint32_t samplesUsed = 0;
        uint32_t numSamples = std::min(m_samplesPerIteration, m_trainingSamples);

        for (int i=0; (samplesUsed < m_trainingSamples || m_trainingMaxSeconds) && !m_canceled; ++i)
        {
            const bool trustGuidingCaches = (m_guidingField.m_totalSPP >= m_minSPPToTrustGuidingCaches);

            m_adrr                          =  trustGuidingCaches && adrr();
            m_accountForDirectLightMiWeight =  trustGuidingCaches && accountForDirectLightMiWeight();
            m_rrDepth                       = (trustGuidingCaches) ? rrDepth() : m_maxDepth;

            if (numSamples < m_trainingSamples-samplesUsed || m_trainingMaxSeconds)
            {
                Log(EInfo, "rendering training iteration %i with %ispp", i, numSamples);
            }
            else
            {
                numSamples = m_trainingSamples-samplesUsed;
                Log(EInfo, "rendering training iteration %i with %ispp (final training iteration)", i, numSamples);
            }

            trainingSamplerProps.removeProperty("sampleCount");
            trainingSamplerProps.setSize("sampleCount", numSamples);
            if (m_deterministic)
            {
                trainingSamplerProps.removeProperty("salt");
                trainingSamplerProps.setSize("salt", m_guidingField.m_totalSPP+m_postponedSPPOfData);
            }
            ref<Sampler> trainingSampler = static_cast<Sampler*>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), trainingSamplerProps));
            trainingSampler->configure();

            trainingScene->setSampler(trainingSampler);

            /* Create a sampler instance for every core */
            std::vector<SerializableObject *> samplers(sched->getCoreCount());
            for (size_t i=0; i<sched->getCoreCount(); ++i) {
                ref<Sampler> clonedSampler = trainingSampler->clone();
                clonedSampler->incRef();
                samplers[i] = clonedSampler.get();
            }

            int trainingSamplerResID = sched->registerMultiResource(samplers);

            for (size_t i=0; i<sched->getCoreCount(); ++i)
                samplers[i]->decRef();

            iterationPreprocess(film);
            SamplingIntegrator::render(trainingScene, queue, job, trainingSceneResID, sensorResID, trainingSamplerResID);
            iterationPostprocess(film, numSamples, job);

            sched->unregisterResource(trainingSamplerResID);

            const uint32_t currentIterationNumSamples = numSamples;
            samplesUsed += currentIterationNumSamples;

            if (m_expGrowSamplesPerIteration)
            {
                numSamples = numSamples<<1;
                // add remaining samples to the last iteration when they run out
                if (m_trainingSamples-samplesUsed-numSamples < numSamples)
                    numSamples = m_trainingSamples-samplesUsed;
            }

            if (m_trainingMaxSeconds)
            {
                const Float iterationSeconds = trainingTimer->lap();
                const Float nextIterationSecondsEstimate = iterationSeconds*static_cast<Float>(numSamples)/static_cast<Float>(currentIterationNumSamples);
                const Float seconds = trainingTimer->getSeconds();
                if (seconds+nextIterationSecondsEstimate >= m_trainingMaxSeconds)
                    break;
            }
        }

        Log(EInfo, "training phase finished after %u samples per pixel in %s.", m_trainingSamples, timeString(trainingTimer->getMilliseconds()/1000.0f, true).c_str());

    #ifndef GUIDING_DETAILED_STATISTICS
        if (m_guidingField.m_iteration)
        {
            GuidingTreeType::Statistics treeStats{m_guidingField.m_guidingTree};
            Log(EInfo, treeStats.toString<PathGuidingSampleData>().c_str());
        }
    #endif

        sched->unregisterResource(trainingSceneResID);

        if (m_exportFinalTree)
        {
            const std::string treeFile = m_trainingDirectory+"final_guiding_field.serialized";

            ref<FileStream> treeSerializationStream = new FileStream(treeFile, FileStream::ETruncWrite);
            m_guidingField.serialize(treeSerializationStream);
            treeSerializationStream->close();
            Log(EInfo, "final guiding field written to %s", treeFile.c_str());
        }
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID)
    {
        ref<Timer> renderTimer = new Timer{};

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        if (m_deterministic)
        {
            Log(EInfo, "forcing the use of the deterministic sampler for rendering");

            Properties deterministicSamplerProps = scene->getSampler()->getProperties();
            deterministicSamplerProps.setPluginName("deterministic");
            ref<Sampler> deterministicSampler = static_cast<Sampler*>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), deterministicSamplerProps));
            deterministicSampler->configure();

            /* Create a sampler instance for every core */
            std::vector<SerializableObject *> deterministicSamplers(sched->getCoreCount());
            for (size_t i=0; i<sched->getCoreCount(); ++i) {
                ref<Sampler> clonedSampler = deterministicSampler->clone();
                clonedSampler->incRef();
                deterministicSamplers[i] = clonedSampler.get();
            }

            samplerResID = sched->registerMultiResource(deterministicSamplers);

            for (size_t i=0; i<sched->getCoreCount(); ++i)
                deterministicSamplers[i]->decRef();

            scene->setSampler(deterministicSampler);
        }

        size_t nCores = sched->getCoreCount();
        const Sampler *sampler = scene->getSampler();
        size_t sampleCount = sampler->getSampleCount();

        if (m_renderMaxSeconds == 0)
            Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
                " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
                sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
                nCores == 1 ? "core" : "cores");
        else
            Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
                " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
                m_renderMaxSeconds, m_renderMaxSeconds == 1 ? "second" : "seconds", nCores,
                nCores == 1 ? "core" : "cores");

        int integratorResID = sched->registerResource(this);

        Float totalRenderTime;
        Float iterationRenderTime;
        ParallelProcess::EStatus status;
        size_t iterations = 0;
        do
        {
            if (m_deterministic)
            {
                for (size_t i=0; i<nCores; ++i)
                {
                    Sampler *sampler = static_cast<Sampler *>(sched->getResource(samplerResID, i));
                    sampler->setSampleIndex(iterations*sampleCount*sampleCount);
                }
            }

            /* This is a sampling-based integrator - parallelize */
            ref<ParallelProcess> proc = new BlockedRenderProcess(job, queue, scene->getBlockSize());
            proc->bindResource("integrator", integratorResID);
            proc->bindResource("scene", sceneResID);
            proc->bindResource("sensor", sensorResID);
            proc->bindResource("sampler", samplerResID);
            scene->bindUsedResources(proc);
            bindUsedResources(proc);

            sched->schedule(proc);

            m_process = proc;
            sched->wait(proc);
            m_process = nullptr;
            status = proc->getReturnStatus();

            iterationRenderTime = renderTimer->lap();
            totalRenderTime = renderTimer->getSeconds();
            ++iterations;

            if (m_exportIntermediateRenders)
            {
                const std::string progressiveRenderFileName = m_trainingDirectory+"render_"
                        +std::to_string(iterations*sampler->getSampleCount())+"spp.exr";

                ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat32, film->getSize());
                film->develop(film->getCropOffset(), film->getSize(), film->getCropOffset(), bitmap);
                bitmap->write(progressiveRenderFileName);
            }
        }
        while (status == ParallelProcess::ESuccess && totalRenderTime+iterationRenderTime < m_renderMaxSeconds);

        Log(EInfo, "rendered %zu samples per pixel in %s.", iterations*sampler->getSampleCount(), timeString(renderTimer->getMilliseconds()/1000.0f, true).c_str());

        if (m_deterministic)
            sched->unregisterResource(samplerResID);
        sched->unregisterResource(integratorResID);

        return status == ParallelProcess::ESuccess;
    }

    void iterationPreprocess(ref<Film> film)
    {
        film->clear();
        Statistics::getInstance()->resetAll();

        m_timer->reset();
    }

    void iterationPostprocess(const ref<Film> film, uint32_t numSPP, const RenderJob *job)
    {
        const Float renderTime = m_timer->stop();
        Log(EInfo, "iteration render time: %s", timeString(renderTime, true).c_str());

        if (m_training && !m_canceled)
        {
            m_timer->reset();

            const size_t numValidSamples = std::accumulate(m_perThreadSamples.begin(), m_perThreadSamples.end(), 0UL,
                                                        [](size_t sum, const IterableBlockedVector<PathGuidingSampleData>& samples) -> size_t {return sum+samples.size();});

            if (EXPECT_TAKEN(numValidSamples >= m_minSamplesToStartFitting))
            {
                if (m_postponedSPPOfData)
                {
                    numSPP += m_postponedSPPOfData;
                    m_postponedSPPOfData = 0;
                }

                ref<Timer> sampleAccumulationTimer = new Timer();
                IterableBlockedVector<PathGuidingSampleData> samples{m_perThreadSamples};
                IterableBlockedVector<Point> zeroValuedSamples{m_perThreadZeroValuedSamples};
                Log(EInfo, "got %zu samples with non-zero radiance and %zu zero-valued samples.", samples.size(), zeroValuedSamples.size());
                Log(EInfo, "sample accumulation time: %s", timeString(sampleAccumulationTimer->getSeconds(), true).c_str());

                if (m_deterministic)
                    std::sort(samples.begin(), samples.end());

                if (m_guidingField.m_iteration == 0 || !m_update)
                    m_guidingField.initField(m_sceneSpace);
                m_guidingField.updateField(samples, zeroValuedSamples, job);
                m_guidingField.addTrainingIteration(numSPP);

    #ifdef GUIDING_DETAILED_STATISTICS
                {
                    GuidingTreeType::Statistics treeStats{m_guidingField.m_guidingTree};
                    Log(EInfo, treeStats.toString<PathGuidingSampleData>().c_str());
                }
    #endif
                Log(EInfo, toString().c_str());

                const Float postprocessingTime = m_timer->stop();
                Log(EInfo, "iteration postprocessing time: %s", timeString(postprocessingTime, true).c_str());
                m_timer->reset();

                const std::string exportFileBasename = m_trainingDirectory+std::to_string(m_guidingField.m_totalSPP)+"spp_total";

                if (m_exportIntermediateRenders)
                {
                    ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat32, film->getSize());
                    film->develop(film->getCropOffset(), film->getSize(), film->getCropOffset(), bitmap);
                    bitmap->write(exportFileBasename+".exr");
                }

                if (m_exportIntermediateSamples)
                {
                    const std::string sampleFile = exportFileBasename+"_samples.serialized";


                    ref<FileStream> sampleSerializationStream = new FileStream(sampleFile, FileStream::ETruncWrite);
                    const size_t numSamples = samples.size();
                    ProgressReporter exportProgress("exporting samples", numSamples, job);
                    sampleSerializationStream->writeSize(numSamples);
                    for (size_t i=0; i<numSamples; ++i)
                    {
                        sampleSerializationStream->write(&samples[i], sizeof(PathGuidingSampleData));
                        exportProgress.update(i);
                    }

                    const size_t streamSize = sampleSerializationStream->getSize();
                    exportProgress.finish();
                    Log(EInfo, "serialized %s of sample data.", memString(streamSize).c_str());
                    sampleSerializationStream->close();

                    Log(EInfo, "samples written to %s", sampleFile.c_str());
                }

                if (m_exportIntermediateTrees)
                {
                    const std::string treeFile = exportFileBasename+"_guiding_field.serialized";

                    ref<FileStream> treeSerializationStream = new FileStream(treeFile, FileStream::ETruncWrite);
                    m_guidingField.serialize(treeSerializationStream);
                    treeSerializationStream->close();
                    Log(EInfo, "guiding field written to %s", treeFile.c_str());
                }

                const Float dataExportTime = m_timer->stop();
                Log(EInfo, "iteration data export time: %s", timeString(dataExportTime, true).c_str());
            }
            else
            {
                m_postponedSPPOfData += numSPP;
                Log(EInfo, "skipped fit due to insufficient sample data (got %zu/%zu valid samples).", numValidSamples, m_minSamplesToStartFitting);
            }
        }

    #ifdef GUIDING_DETAILED_STATISTICS
        //only the base of avgPathLength is incremented by Li
        const size_t numPaths = avgPathLength.getBase();
        //update the base of the remaining statistics here to reduce the load on the cache
        pathsTerminatedByRussianRoulette.incrementBase(numPaths);
        pathsTerminatedByZeroReflectance.incrementBase(numPaths);
        pathsTerminatedByInvalidBSDFDirectionSample.incrementBase(numPaths);
        pathsTerminatedByInvalidGuidedDirectionSample.incrementBase(numPaths);
        pathsTerminatedByMaxDepth.incrementBase(numPaths);
        pathsTerminatedByNoIntersection.incrementBase(numPaths);
        pathsNeverIntersectingTheScene.incrementBase(numPaths);
        zeroPaths.incrementBase(numPaths);
        //update base of secondary counters
        guidedDirectionSamples.incrementBase(avgPathLength.getValue());
        zeroPathLengths.incrementBase(zeroPaths.getValue());
        pathsGuidedOutOfScene.incrementBase(pathsTerminatedByNoIntersection.getValue()-pathsNeverIntersectingTheScene.getValue());
    #endif

        Statistics::getInstance()->printStats();
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID) {

        m_canceled = false;

        if (m_renderMaxSeconds && scene->getSampler()->getSampleCount() > 4)
            Log(EError, "For progressive rendering, the sample count must be set to at most 4.\n(A too high value would terminate the rendering process too early.)");

        if (m_useBSDFProduct)
            BSDFOracle::initBSDFRepresentations(scene);

        m_sceneSpace = scene->getKDTree()->getAABB();
        m_envmapDistance = scene->getAABB().getBSphere().radius*1024.0f;

        //set up per thread storage of sample data
        ref<Scheduler> sched = Scheduler::getInstance();
        const size_t nCores = sched->getCoreCount();
        m_perThreadIntersectionData.resize(nCores);
        m_perThreadSamples.resize(nCores);
        m_perThreadZeroValuedSamples.resize(nCores);
        m_maxThreadId.store(0);

        bool importedGuidingField = false;

        if (!m_importTreeFile.empty())
        {
            const fs::path guidingFieldImportPath = m_trainingDirectory+m_importTreeFile;
            if (fs::exists(guidingFieldImportPath))
            {
                ref<FileStream> guidingFieldImportStream = new FileStream(guidingFieldImportPath, FileStream::EReadOnly);
                m_guidingField = GuidingFieldType{guidingFieldImportStream};
                importedGuidingField = true;
                guidingFieldImportStream->close();
                Log(EInfo, "guiding field imported from %s", guidingFieldImportPath.c_str());
            }
            else
                Log(EWarn, "Serialized GuidingField could not be found! (%s)", guidingFieldImportPath.c_str());
        }

        //reset the guiding field
        if (!importedGuidingField)
            m_guidingField = GuidingFieldType{m_guidingFieldProps};

        Log(EInfo, m_guidingField.toString().c_str());

        m_postponedSPPOfData = 0;

        if (m_trainingSamples || m_trainingMaxSeconds)
            train(static_cast<Scene*>(sched->getResource(sceneResID)), queue, job, sensorResID);

        if (m_canceled)
            return false;

        if (m_disableParallaxAfterTraining)
        {
            Log(EWarn, "disabling parallax, as requested.");
            m_guidingField.m_factory.m_statisticsFactory.m_parallaxCompensation = false;
        }

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        iterationPreprocess(sensor->getFilm());

        return true;
    }

    void postprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID)
    {
        (void)queue; (void)sceneResID; (void)samplerResID;

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        iterationPostprocess(film, static_cast<uint32_t>(scene->getSampler()->getSampleCount()), job);

        m_perThreadIntersectionData.clear();
        m_perThreadSamples.clear();
        m_perThreadZeroValuedSamples.clear();

        Log(EInfo, this->toString().c_str());
        Log(EInfo, m_guidingField.toString().c_str());
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        RayDifferential ray(r);

        if (EXPECT_NOT_TAKEN(m_renderIndirectLight))
            rRec.type &= ~(RadianceQueryRecord::EDirectSurfaceRadiance|RadianceQueryRecord::EEmittedRadiance);

        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
        intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum Li {0.0f};
        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        auto miWeight = m_usePowerHeuristic ? [](Float pdfA, Float pdfB) -> Float
        {
            pdfA *= pdfA;
            pdfB *= pdfB;
            return pdfA / (pdfA + pdfB);
        } : [](Float pdfA, Float pdfB) -> Float
        {
            return pdfA / (pdfA + pdfB);
        };

        static thread_local std::vector<PathGuidingIntersectionData>* intersectionData {nullptr};
        static thread_local IterableBlockedVector<PathGuidingSampleData>* samples {nullptr};
        static thread_local IterableBlockedVector<Point>* zeroValuedSamples {nullptr};
        static thread_local GuidingFieldType::GuidedBSDFType guidedBSDF{m_bsdfProbability, m_useCosineProduct, m_useBSDFProduct};
        //initialize on first use
        if (EXPECT_NOT_TAKEN(m_training && !m_threadLocalInitialized.get()))
        {
            const size_t threadId = m_maxThreadId.fetch_add(1, std::memory_order_relaxed);
            intersectionData = &m_perThreadIntersectionData.at(threadId);
            samples = &m_perThreadSamples.at(threadId);
            zeroValuedSamples = &m_perThreadZeroValuedSamples.at(threadId);
            guidedBSDF = GuidingFieldType::GuidedBSDFType{m_bsdfProbability, m_useCosineProduct, m_useBSDFProduct};

            m_threadLocalInitialized.get() = true;
        }

        struct ADRRData
        {
            Spectrum measurementEstimate{0.0f};
            Spectrum diffuseReflectionEstimate{0.0f};
            Spectrum reflectedRadianceEstimate{0.0f};
            Spectrum incidentRadianceEstimate{0.0f};
            bool canApproximateCurrentSurfaceAsDiffuse{false};
        } adrrData;

        PathGuidingIntersectionData dummyIntersectionData{rRec.its};

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0)
        {
            if (m_training)
                intersectionData->emplace_back(rRec.its);
            PathGuidingIntersectionData& currentIntersectionData = m_training ? intersectionData->back() : dummyIntersectionData;

            if (!rRec.its.isValid())
            {
                /* If no intersection could be found, potentially return
                radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                {
                    const Spectrum envmapRadiance = rRec.scene->evalEnvironment(ray);
                    Li += throughput * envmapRadiance;

                    currentIntersectionData.emission = ContributionAndThroughput{envmapRadiance, Spectrum{1.0f}};
                }

    #ifdef GUIDING_DETAILED_STATISTICS
                ++pathsNeverIntersectingTheScene;
                ++pathsTerminatedByNoIntersection;
    #endif
                break;
            }

            const BSDF *surfaceBSDF = rRec.its.getBSDF(ray);

            m_guidingField.configureGuidedBSDF(guidedBSDF, rRec.its, -ray.d, surfaceBSDF);

            const bool canUseGuiding = guidedBSDF.canUseGuiding();

            if (canUseGuiding)
            {
                if (EXPECT_NOT_TAKEN(m_renderNumComponents))
                    return Spectrum{static_cast<Float>(guidedBSDF.getNumMixtureComponents())};

                if (m_splatSamples)
                {
                    currentIntersectionData.regionBounds = guidedBSDF.getRegionBounds();
                    currentIntersectionData.splattingVolume = guidedBSDF.getSampleBounds();
                }
            }

            /* Possibly include emitted radiance if requested */
            if (rRec.its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
            {
                const Spectrum emittedRadiance = rRec.its.Le(-ray.d);
                Li += throughput * emittedRadiance;

                currentIntersectionData.emission.contribution = emittedRadiance;
            }

            /* Include radiance from a subsurface scattering model if requested */
            if (rRec.its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
            {
                const Spectrum subsurfaceScatteredRadiance = rRec.its.LoSub(rRec.scene, rRec.sampler, -ray.d, rRec.depth);
                Li += throughput * subsurfaceScatteredRadiance;

                currentIntersectionData.emission.contribution += subsurfaceScatteredRadiance;
            }

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, rRec.its.geoFrame.n)
                    * Frame::cosTheta(rRec.its.wi) >= 0)) {

                /* Only continue if:
                1. The current path length is below the specifed maximum
                2. If 'strictNormals'=true, when the geometric and shading
                    normals classify the incident direction to the same side */

    #ifdef GUIDING_DETAILED_STATISTICS
                ++pathsTerminatedByMaxDepth;
    #endif
                break;
            }


            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(rRec.its);

            if (m_useNee && (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (guidedBSDF.getType() & BSDF::ESmooth))) {
                Spectrum value = rRec.scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(rRec.its, rRec.its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = guidedBSDF.eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                            || dot(rRec.its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                        using PathGuiding / BSDF sampling */
                        const Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                                ? guidedBSDF.pdf(bRec) : 0;

                        /* Weight using the power heuristic */
                        const Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * weight;

                        currentIntersectionData.neeDirectLight = ContributionAndThroughput{value, bsdfVal*weight};
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(rRec.its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = guidedBSDF.sample(bRec, bsdfPdf, rRec.nextSample2D());

    #ifdef GUIDING_DETAILED_STATISTICS
            if (bRec.sampledType & BSDF::EGuiding)
                ++guidedDirectionSamples;
    #endif

            if (bsdfWeight.isZero())
            {
    #ifdef GUIDING_DETAILED_STATISTICS
                if (bRec.sampledType == BSDF::EGuiding)
                    ++pathsTerminatedByInvalidGuidedDirectionSample;
                else if (guidedBSDF.hasComponent(BSDF::EAll))
                    ++pathsTerminatedByInvalidBSDFDirectionSample;
                else
                    ++pathsTerminatedByZeroReflectance;
    #endif
                break;
            }

            scattered |= bRec.sampledType != BSDF::ENull;

            if (m_adrr)
            {
                adrrData.reflectedRadianceEstimate = adrrData.incidentRadianceEstimate;

                //approximate non-transmissive surfaces as diffuse, starting from the second bounce
                adrrData.canApproximateCurrentSurfaceAsDiffuse = canUseGuiding && (((bRec.sampledType&BSDF::ESmooth&BSDF::EReflection) && rRec.depth > 1) || !surfaceBSDF->hasComponent(BSDF::EAll&~BSDF::EDiffuseReflection));

                if (canUseGuiding)
                {
                    if (adrrData.canApproximateCurrentSurfaceAsDiffuse)
                        adrrData.diffuseReflectionEstimate = guidedBSDF.estimateReflectedRadiance(bRec, rRec.its);

                    adrrData.incidentRadianceEstimate = guidedBSDF.estimateIncidentRadiance(bRec.its.toWorld(bRec.wo));
                }
                //on specular surfaces, the incident radiance estimate remains unchanged
            }

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = rRec.its.toWorld(bRec.wo);

            currentIntersectionData.wiWorld = wo;
            currentIntersectionData.pdfWiWorld = bsdfPdf;
            currentIntersectionData.cosThetaI = bRec.wo.z;
            currentIntersectionData.sampledType = bRec.sampledType;

            ++rRec.depth;

            Float woDotGeoN = dot(rRec.its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(rRec.its.p, wo, ray.time);
            if (rRec.scene->rayIntersect(ray, rRec.its)) {
                /* Intersected something - check if it was a luminaire */
                if (rRec.its.isEmitter()) {
                    value = rRec.its.Le(-ray.d);
                    dRec.setQuery(ray, rRec.its);
                    hitEmitter = true;
                }
            } else {
    #ifdef GUIDING_DETAILED_STATISTICS
                ++pathsTerminatedByNoIntersection;
                if (bRec.sampledType & BSDF::EGuiding)
                    ++pathsGuidedOutOfScene;
    #endif

                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = rRec.scene->getEnvironmentEmitter();

                if (env) {
                    if (m_hideEmitters && !scattered)
                        break;

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;

                    hitEmitter = true;
                } else {
                    break;
                }
            }

            /* Keep track of the throughput and relative
            refractive index along the path */
            const Spectrum liBeforeBSDF = Li;
            const Spectrum throughputBeforeBSDF = throughput;
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            currentIntersectionData.throughputFactors = bsdfWeight;
            currentIntersectionData.eta = bRec.eta;
            currentIntersectionData.roughness = surfaceBSDF->getRoughness(rRec.its, bRec.sampledComponent);

            /* If a luminaire was hit, estimate the local illumination and
            weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                implemented direct illumination sampling technique */
                const Float lumPdf = (m_useNee && !(bRec.sampledType & BSDF::EDelta)) ?
                    rRec.scene->pdfEmitterDirect(dRec) : 0;
                Float weight = (m_useNee) ? miWeight(bsdfPdf, lumPdf) : 1.0f;
                Li += throughput * value * weight;

                currentIntersectionData.bsdfDirectLight = ContributionAndThroughput{value, bsdfWeight*weight};
                currentIntersectionData.bsdfMiWeight = weight;
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
            BSDF sample or if indirect illumination was not requested */
            if (!rRec.its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (m_adrr)
            {
                //estimate the remaining path contribution for ADRR by approximating the diffuse reflectance of the current surface
                if (adrrData.canApproximateCurrentSurfaceAsDiffuse && adrrData.measurementEstimate.isZero())
                {
                    //using the conservative adjoint
                    Spectrum adjoint;
                    adjoint[0] = std::max(adrrData.reflectedRadianceEstimate[0], adrrData.diffuseReflectionEstimate[0]);
                    adjoint[1] = std::max(adrrData.reflectedRadianceEstimate[1], adrrData.diffuseReflectionEstimate[1]);
                    adjoint[2] = std::max(adrrData.reflectedRadianceEstimate[2], adrrData.diffuseReflectionEstimate[2]);

                    if (!m_guideDirectLight || (m_useNee && m_accountForDirectLightMiWeight))
                        adrrData.measurementEstimate = Li+throughputBeforeBSDF*adjoint;
                    else
                        adrrData.measurementEstimate = liBeforeBSDF+throughputBeforeBSDF*adjoint;

                    if (m_renderMeasurementEstimate)
                        return adrrData.measurementEstimate;
                }

                //remove already gathered radiance from the remaining path contribution estimate
                if (m_guideDirectLight && hitEmitter)
                    adrrData.incidentRadianceEstimate -= currentIntersectionData.bsdfDirectLight.contribution*currentIntersectionData.bsdfMiWeight;
            }

            if (m_rrDepth >= 0 && rRec.depth > m_rrDepth)
            {
                Float q = 0.95f;

                if (m_adrr)
                {
                    if (adrrData.canApproximateCurrentSurfaceAsDiffuse && !adrrData.measurementEstimate.isZero())
                    {
                        //using the conservative adjoint
                        Spectrum adjoint;
                        adjoint[0] = std::max(adrrData.reflectedRadianceEstimate[0], adrrData.diffuseReflectionEstimate[0]);
                        adjoint[1] = std::max(adrrData.reflectedRadianceEstimate[1], adrrData.diffuseReflectionEstimate[1]);
                        adjoint[2] = std::max(adrrData.reflectedRadianceEstimate[2], adrrData.diffuseReflectionEstimate[2]);

                        if (adjoint.isValid())
                        {
                            const Spectrum center = adrrData.measurementEstimate/adjoint;
                            const float s = 5.0f;
                            const Spectrum min = 2.0f*center/(1.0f+s);
                            //const float max = s*min;

                            if (PATHGUIDING_SPECTRUM_TO_FLOAT(throughputBeforeBSDF-min) < 0.0f)
                            {
                                q = std::min(std::max(0.1f, PATHGUIDING_SPECTRUM_TO_FLOAT(throughputBeforeBSDF/min)), q);
                            }
                            //if (minThroughput > max)
                            //{
                            //    //TODO: split
                            //}
                        }
                    }
                }
                else
                {
                    /* Russian roulette: try to keep path weights equal to one,
                    while accounting for the solid angle compression at refractive
                    index boundaries. Stop with at least some probability to avoid
                    getting stuck (e.g. due to total internal reflection) */

                    q = std::min(throughput.max() * eta * eta, q);
                }

                if (rRec.nextSample1D() >= q)
                {
    #ifdef GUIDING_DETAILED_STATISTICS
                    ++pathsTerminatedByRussianRoulette;
    #endif
                    if (m_training && !currentIntersectionData.bsdfDirectLight.contribution.isZero())
                        intersectionData->emplace_back(rRec.its);
                    break;
                }

                const Float invQ = 1.0f/q;

                throughput *= invQ;
                currentIntersectionData.throughputFactors *= invQ;
            }
        }

        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        if (EXPECT_NOT_TAKEN(m_renderNumComponents))
            return Spectrum{0.0f};

    #ifdef GUIDING_DETAILED_STATISTICS
        if (Li.isZero())
        {
            ++zeroPaths;
            zeroPathLengths += rRec.depth;
        }
    #endif

        if (m_training)
        {
            float currentDistance {rRec.its.isValid() ? 0.0f : m_envmapDistance};
            float lastDistance {m_envmapDistance};

    #ifdef GUIDING_DETAILED_STATISTICS
            size_t numSamples {0}, numZeroSamples{0};
    #endif

            //generate samples by traversing the path back to front
            for (int i=intersectionData->size()-1; i>=0; --i)
            {
                const PathGuidingIntersectionData& currentIntersectionData = (*intersectionData)[i];

                if ((currentIntersectionData.sampledType&BSDF::ESmooth) && currentIntersectionData.roughness > 0.01f)
                {
    #ifdef GUIDING_DETAILED_STATISTICS
                    ++numSamples;
    #endif
                    Point pos = currentIntersectionData.pos;
                    Vector wiWorld = currentIntersectionData.wiWorld;
                    float distance = currentDistance;

                    uint32_t flags {0};

                    if (m_splatSamples && !currentIntersectionData.splattingVolume.isEmpty())
                    {
                        //start with a uniform direction vector, scaled by a uniform radius, favoring points close to the center
                        //(deliberately not uniformly sampling positions within the sphere's volume!)
                        const Vector randomDirLowDistance = warp::squareToUniformSphere(Point2{rRec.sampler->next1D(), rRec.sampler->next1D()})*rRec.sampler->next1D();
                        //scale with the splatting volume's extents
                        Vector splattingExtents = currentIntersectionData.splattingVolume.getExtents()*0.5f;
                        //limit displacement distance to last travelled distance
                        const float splattingMaxRadius = (splattingExtents.x > splattingExtents.y) ? (splattingExtents.x > splattingExtents.z ? splattingExtents.x : splattingExtents.z)
                                                                                                : (splattingExtents.y > splattingExtents.z ? splattingExtents.y : splattingExtents.z);
                        if (splattingMaxRadius > lastDistance)
                            splattingExtents *= lastDistance/splattingMaxRadius;
                        const Vector displacement {randomDirLowDistance.x*splattingExtents.x, randomDirLowDistance.y*splattingExtents.y, randomDirLowDistance.z*splattingExtents.z};
                        Point newPos {pos+displacement};

                        //only move the sample if the new pos is outside of the region's bounding box
                        if ((!currentIntersectionData.regionBounds.contains(newPos)) && !std::isinf(distance) && distance > 0.0f)
                        {
                            const Point pLight = pos+wiWorld*distance;
                            //enforce sample position within scene bounds
                            for (int i=0; i<Point::dim; ++i)
                            {
                                if (newPos[i] < m_sceneSpace.min[i])
                                    newPos[i] = m_sceneSpace.min[i];
                                if (newPos[i] > m_sceneSpace.max[i])
                                    newPos[i] = m_sceneSpace.max[i];
                            }

                            pos = newPos;
                            wiWorld = pLight-newPos;
                            distance = wiWorld.length();
                            wiWorld /= distance;

                            flags |= Guiding::ESplatted;
                        }
                    }

                    Spectrum clampedLight {0.0f};
                    const float maxThroughput = 10.0f;
                    const float minPDF = 0.1f;

                    for (size_t j=i+1; j<intersectionData->size(); ++j)
                    {
                        clampedLight += (*intersectionData)[j].bsdfDirectLight.getClamped(maxThroughput);
                        clampedLight += (*intersectionData)[j].neeDirectLight.getClamped(maxThroughput);
                        clampedLight += (*intersectionData)[j].emission.getClamped(maxThroughput);
                    }

                    if (m_guideDirectLight)
                    {
                        if (!m_useNee || !m_accountForDirectLightMiWeight)
                            clampedLight += currentIntersectionData.bsdfDirectLight.contribution;
                        else
                            clampedLight += currentIntersectionData.bsdfDirectLight.contribution*currentIntersectionData.bsdfMiWeight;
                    }

                    if (clampedLight.isZero())
                    {
    #ifdef GUIDING_DETAILED_STATISTICS
                        ++numZeroSamples;
    #endif
                        if (m_collectZeroValuedSamples)
                            zeroValuedSamples->push_back(pos);
                    }
                    else
                    {
                        const float clampedPDF = std::max(minPDF, currentIntersectionData.pdfWiWorld);
                        samples->emplace_back(pos, wiWorld, PATHGUIDING_SPECTRUM_TO_FLOAT(clampedLight)/clampedPDF,
                                            clampedPDF, distance, flags);
                    }
                }
                //NOTE: this should probably be a bit more gradual in between (full accumulation of the distance on speccular <--> no accumulation of distance on diffuse)
                if ((currentIntersectionData.sampledType&BSDF::ESmooth) && currentIntersectionData.roughness >= 0.3f)
                {
                    currentDistance = currentIntersectionData.distance;
                }
                else
                {
                    if (currentIntersectionData.eta != 1.0f)
                        currentDistance *= fabs(currentIntersectionData.cosThetaO/(currentIntersectionData.cosThetaI*currentIntersectionData.eta));

                    currentDistance += currentIntersectionData.distance;
                }

                lastDistance = currentIntersectionData.distance;

                if (i == 0)
                    break;

                //update throughput of following entries
                for (size_t j=i+1; j<intersectionData->size(); ++j)
                    (*intersectionData)[j] *= currentIntersectionData.throughputFactors;
            }
    #ifdef GUIDING_DETAILED_STATISTICS
            zeroSamples.incrementBase(numSamples);
            zeroSamples += numZeroSamples;
    #endif

            intersectionData->clear();
        }

        return Li;
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "PathGuiding[\n"
            << "  path tracing configuration:\n"
            << "    maxDepth: " << m_maxDepth << '\n'
            << "    ";
        if (m_strictNormals)
            oss << "strictNormals ";
        if (m_hideEmitters)
            oss << "hideEmitters ";
        if (m_usePowerHeuristic)
            oss << "usePowerHeuristic ";
        if (m_useNee)
            oss << "useNee ";
        if (m_useCosineProduct)
            oss << "useCosineProduct ";
        if (m_useBSDFProduct)
            oss << "useBSDFProduct ";
        if (m_guideDirectLight)
            oss << "guideDirectLight ";
        if (m_renderMeasurementEstimate)
            oss << "renderMeasurementEstimate ";
        if (m_renderNumComponents)
            oss << "renderNumComponents ";
        if (m_renderIndirectLight)
            oss << "renderIndirectLight ";

        oss << '\n'
            << "  rr configuration:\n"
            << "    rrDepth: "  << m_rrDepth  << '\n'
            << "    ";
        if (m_adrr)
            oss << "adrr ";

        oss << '\n'
            << "  guiding configuration:\n"
            << "    bsdfProbability: "  << m_bsdfProbability << '\n'
            << "    minSPPToTrustGuidingCaches: "  << m_minSPPToTrustGuidingCaches << '\n'
            << "    minSamplesToStartFitting: "  << m_minSamplesToStartFitting << '\n'
            << "    ";
        if (m_splatSamples)
            oss << "splatSamples ";
        if (m_accountForDirectLightMiWeight)
            oss << "accountForDirectLightMiWeight ";
        if (m_update)
            oss << "update ";
        if (m_deterministic)
            oss << "deterministic ";

        oss << '\n'
            << "  import/export:\n";
        if (!m_importTreeFile.empty())
            oss << "    importTreeFile: " << m_importTreeFile << '\n';
        oss << "    trainingDirectory: " << m_trainingDirectory << '\n'
            << "    ";
        if (m_exportIntermediateRenders)
            oss << "exportIntermediateRenders ";
        if (m_exportIntermediateTrees)
            oss << "exportIntermediateTrees ";
        if (m_exportIntermediateSamples)
            oss << "exportIntermediateSamples ";
        if (m_exportFinalTree)
            oss << "exportFinalTree ";

        oss << '\n'
            << "]\n";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};
MTS_IMPLEMENT_CLASS_S(PathGuiding, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathGuiding, "Robust Fitting of Parallax-Aware Mixtures for Path Guiding")
MTS_NAMESPACE_END