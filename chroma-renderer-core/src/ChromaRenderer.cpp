#include "ChromaRenderer.h"

#include <fstream>
#include <iostream>

#include "Config.h"
#include "ModelImporter.h"
#include <RTUtils.h>
#include <functional>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TILESIZE 64

ChromaRenderer::ChromaRenderer() : state(State::IDLE)
{
    setSettings(settings);
}
ChromaRenderer::~ChromaRenderer()
{
    // Need to stop all workers before releasing memory (especially the image object).
    threadPool.abort();
}
void ChromaRenderer::genTasks()
{
    const unsigned int width = scene.camera.width;
    const unsigned int height = scene.camera.height;

    // Divide the screen in squared tiles of approximately TILESIZE pixels wide
    const int widthDivs = (int)floor(width / TILESIZE);
    const int heightDivs = (int)floor(height / TILESIZE);

    const int widthStep = static_cast<int>(std::ceil(width / (float)widthDivs));
    const int heightStep = static_cast<int>(std::ceil(height / (float)heightDivs));

    // Put all tiles in the queue
    // std::vector<Interval> tasks;
    // tasks.clear();
    for (int i = 0; i < widthDivs; i++)
    {
        Interval interval;
        for (int j = 0; j < heightDivs; j++)
        {
            interval.fromWidth = widthStep * i;
            interval.fromHeight = heightStep * j;
            if (i == widthDivs - 1 && j == heightDivs - 1)
            {
                interval.toWidth = width;
                interval.toHeight = height;
            }
            else if (i == widthDivs - 1)
            {
                interval.toWidth = width;
                interval.toHeight = heightStep * (j + 1);
            }
            else if (j == heightDivs - 1)
            {
                interval.toWidth = widthStep * (i + 1);
                interval.toHeight = height;
            }
            else
            {
                interval.toWidth = widthStep * (i + 1);
                interval.toHeight = heightStep * (j + 1);
            }
            switch (rendererType)
            {
            case ChromaRenderer::RAYCAST:
                threadPool.enqueue(std::bind(&RayCasting::trace,
                                             std::ref(renderer),
                                             std::ref(scene),
                                             std::ref(image),
                                             std::ref(settings),
                                             interval,
                                             std::placeholders::_1));
                break;
            case ChromaRenderer::PATHTRACE:
                threadPool.enqueue(std::bind(&PathTracing::trace,
                                             std::ref(pathtracing),
                                             std::ref(scene),
                                             std::ref(image),
                                             interval,
                                             std::placeholders::_1));
                break;
            default:
                break;
            }
        }
    }

    // std::random_shuffle(tasks.begin(), tasks.end());
}

void ChromaRenderer::setSize(unsigned int width, unsigned int height)
{
    image.setSize(width, height);
    scene.camera.setSize(width, height);
    // scene.camera.computeUVW();
}
void ChromaRenderer::setSettings(RendererSettings& psettings)
{
    settings = psettings;
    scene.camera.horizontalFOV(settings.horizontalFOV);
    setSize(settings.width, settings.height);
    pixelCount = settings.width * settings.height;
    invPixelCount = 1.f / (float)pixelCount;
    threadPool.setNumberWorkers(settings.nthreads);
    pathtracing.setSettings(settings);
    cudaPathTracer.setSettings(settings);
}
RendererSettings ChromaRenderer::getSettings()
{
    return settings;
}
void ChromaRenderer::startRender()
{
    start();
}
void ChromaRenderer::startRender(RendererSettings& psettings)
{
    setSettings(psettings);
    start();
}
void ChromaRenderer::start()
{
    if (!isIdle())
        return;

    // <Init>
    threadPool.clearTaskQueue();
    image.clear();

    switch (rendererType)
    {
    case ChromaRenderer::RAYCAST:
        renderer.init();
        break;
    case ChromaRenderer::PATHTRACE:
        pathtracing.init();
        break;
    case ChromaRenderer::CUDAPATHTRACE:
        cudaPathTracer.init(image, scene.camera);
        break;
    }
    // </Init>

    stopwatch.restart();

    switch (rendererType)
    {
    case ChromaRenderer::RAYCAST:
    case ChromaRenderer::PATHTRACE:
        genTasks();
        break;
    case ChromaRenderer::CUDAPATHTRACE:
        // threadPool.enqueue(std::bind(&CudaPathTracer::renderThread, std::ref(cudaPathTracer), std::ref(image),
        // std::placeholders::_1));
        break;
    }

    state = State::RENDERING;
    running = true;
}
void ChromaRenderer::stopRender(bool /*block*/)
{
    threadPool.clearTaskQueue();
    running = false;
    state = State::IDLE;
}

void ChromaRenderer::importScene(std::string filename)
{
    importScene(filename, []() {});
}

void ChromaRenderer::importScene(std::string filename, std::function<void()> onLoad)
{
    if (!isIdle())
        return;
    /*scene.clear();
    Object o;
    ModelImporter::import(filename, o);
    scene.addObject(o);*/

    /*Mesh m;
    Object o;

    ModelImporter::import(filename, m);
    ModelImporter::import(filename, o);

    if (m.t.size() != o.f.size())
        std::cout << "Different sizes" << std::endl;

    for (int i = 0; i < m.t.size(); i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            if (m.t[i].getVertex(j)->x != o.f[i].v[j].x
                || m.t[i].getVertex(j)->y != o.f[i].v[j].y
                || m.t[i].getVertex(j)->z != o.f[i].v[j].z)
                std::cout << "Different vertex" << std::endl;
        }
    }

    // Compute each triangle's bbox, it's centroid and the bbox of all triangles and of all centroids
    std::vector<BVHPrimitiveInfo> mPrim;
    mPrim.reserve(m.t.size());
    for (int i = 0; i < m.t.size(); i++)
    {
        BoundingBox bb;
        bb = BoundingBox();
        for (int dim = 0; dim < 3; dim++)
            bb.expand(*m.t[i].getVertex(dim));
        mPrim.emplace_back(bb, i);
    }
    // Compute each triangle's bbox, it's centroid and the bbox of all triangles and of all centroids
    std::vector<BVHPrimitiveInfo> oPrim;
    oPrim.reserve(m.t.size());
    for (int i = 0; i < m.t.size(); i++)
    {
        BoundingBox bb;
        bb = BoundingBox();
        for (int dim = 0; dim < 3; dim++)
            bb.expand(*m.t[i].getVertex(dim));
        oPrim.emplace_back(bb, i);
    }


    if (mPrim.size() != oPrim.size())
        std::cout << "Different sizes" << std::endl;

    for (int i = 0; i < mPrim.size(); i++)
    {
        if (mPrim[i].index != oPrim[i].index)
            std::cout << "Different index" << std::endl;
        if (mPrim[i].centroid.x != oPrim[i].centroid.x
            || mPrim[i].centroid.y != oPrim[i].centroid.y
            || mPrim[i].centroid.z != oPrim[i].centroid.z)
            std::cout << "Different centroid" << std::endl;

        if (mPrim[i].bbox.max.x != oPrim[i].bbox.max.x
            || mPrim[i].bbox.max.y != oPrim[i].bbox.max.y
            || mPrim[i].bbox.max.z != oPrim[i].bbox.max.z)
            std::cout << "Different bbox max" << std::endl;

        if (mPrim[i].bbox.min.x != oPrim[i].bbox.min.x
            || mPrim[i].bbox.min.y != oPrim[i].bbox.min.y
            || mPrim[i].bbox.min.z != oPrim[i].bbox.min.z)
            std::cout << "Different bbox min" << std::endl;
    }

    std::cout << "DUDE!" << std::endl;
    std::cin.get();

    return;*/

    /*Mesh m;
    Object o;

    ModelImporter::import(filename, m);
    ModelImporter::import(filename, o);

    std::cout << "uint32_t size:                " << sizeof(uint32_t) << std::endl;
    std::cout << "std::vector<glm::vec3>* size: " << sizeof(std::vector<glm::vec3>*) << std::endl;
    std::cout << "glm::vec3 size:               " << sizeof(glm::vec3) << std::endl;
    std::cout << "Triangle size:                " << sizeof(Triangle) << std::endl;
    std::cout << "Face size:                    " << sizeof(Face) << std::endl;
    std::cout << std::endl;

    std::cout << "Object total size " << o.sizeInBytes() / 1024 << "KB" << std::endl;
    std::cout << "    Faces         " << (sizeof(Face) * o.f.size()) / 1024 << "KB" << "   #" << o.f.size() <<
    std::endl; std::cout << std::endl;

    std::cout << "Mesh   total size " << m.sizeInBytes() / 1024 << "KB" << std::endl;
    std::cout << "    Triangles     " << (sizeof(Triangle) * m.t.size()) / 1024 << "KB" << "   #" << m.t.size() <<
    std::endl; std::cout << "    Vertices      " << (sizeof(glm::vec3) * m.v.size()) / 1024 << "KB" << "   #" <<
    m.v.size() << std::endl; std::cout << "    Normals       " << (sizeof(glm::vec3) * m.n.size()) / 1024 << "KB" << "
    #" << m.n.size() << std::endl;

    std::cin.get();*/

    state = State::LOADINGSCENE;
    // threadPool.enqueue(
    //	std::bind(&ModelImporter::importcbm, filename,
    //	static_cast<std::function<void(Mesh*)>>(std::bind(&ChromaRenderer::cbSceneLoadedm, std::ref(*this),
    // std::placeholders::_1)))
    //	);

    // threadPool.enqueue(
    //	std::bind(&ModelImporter::importcbscene, filename, std::ref(scene),
    //	static_cast<std::function<void()>>(std::bind(&ChromaRenderer::cbSceneLoadedScene, std::ref(*this), onLoad)))
    //	);

    // TODO: load scene asynchronously but only initialize cudaPathTracer (see cbSceneLoadedScene) on the main thread.
    ModelImporter::importcbscene(
        filename,
        scene,
        static_cast<std::function<void()>>(std::bind(&ChromaRenderer::cbSceneLoadedScene, std::ref(*this), onLoad)));
}
void ChromaRenderer::importEnviromentMap(std::string filename)
{
    float* data = nullptr;
    int width, height, channels;
    data = stbi_loadf(filename.c_str(), &width, &height, &channels, 4);

    if (data == nullptr)
        std::cout << "Could not load hdri!" << std::endl;

    std::cout << "Width: " << width << " Height: " << height << " Channels: " << channels << std::endl;

    cudaPathTracer.init(data, width, height);

    stbi_image_free(data);
}
void ChromaRenderer::cbSceneLoadedScene(std::function<void()> onLoad)
{
    state = State::PROCESSINGSCENE;
    scene.sps = new BVH();
    scene.sps->build(scene.meshes);
    cudaPathTracer.init(scene);

    settings.width = scene.camera.width;
    settings.height = scene.camera.height;
    settings.horizontalFOV = scene.camera.horizontalFOV();

    setSettings(settings);

    cudaPathTracer.init(image, scene.camera);

    state = State::IDLE;
    onLoad();
}
void ChromaRenderer::cbSceneLoadedm(Mesh* m)
{
    std::cout << "Mesh   total size " << m->sizeInBytes() / 1024 << "KB" << std::endl;
    std::cout << "    Triangles     " << (sizeof(Triangle) * m->t.size()) / 1024 << "KB"
              << "   #" << m->t.size() << std::endl;
    std::cout << "    Vertices      " << (sizeof(glm::vec3) * m->v.size()) / 1024 << "KB"
              << "   #" << m->v.size() << std::endl;
    std::cout << "    Normals       " << (sizeof(glm::vec3) * m->n.size()) / 1024 << "KB"
              << "   #" << m->n.size() << std::endl;

    scene.clear();
    threadPool.enqueue(std::bind(
        static_cast<void (Scene::*)(Mesh*, std::function<void(void)>)>(&Scene::addMesh),
        std::ref(scene),
        m,
        static_cast<std::function<void(void)>>(std::bind(&ChromaRenderer::cbSceneProcessed, std::ref(*this)))));
    state = State::PROCESSINGSCENE;
}
void ChromaRenderer::cbSceneLoaded(Object o)
{
    scene.clear();
    threadPool.enqueue(std::bind(
        static_cast<void (Scene::*)(Object, std::function<void(void)>)>(&Scene::addObject),
        std::ref(scene),
        o,
        static_cast<std::function<void(void)>>(std::bind(&ChromaRenderer::cbSceneProcessed, std::ref(*this)))));
    state = State::PROCESSINGSCENE;
}
void ChromaRenderer::cbSceneProcessed()
{
    state = State::IDLE;
}

void ChromaRenderer::update()
{
    if (state == State::RENDERING)
    {
        switch (rendererType)
        {
        case ChromaRenderer::RAYCAST:
            break;
        case ChromaRenderer::PATHTRACE:
            break;
        case ChromaRenderer::CUDAPATHTRACE:
            cudaPathTracer.render();
            if (cudaPathTracer.getProgress() >= 1.0f)
            {
                running = false;
                state = State::IDLE;
            }
            break;
        }
    }
}

bool ChromaRenderer::isRunning()
{
    if (running)
    {
        switch (rendererType)
        {
        case ChromaRenderer::RAYCAST:
            running = renderer.donePixelCount < pixelCount;
            break;
        case ChromaRenderer::PATHTRACE:
            running = pathtracing.donePixelCount < pixelCount;
            break;
        case ChromaRenderer::CUDAPATHTRACE:
            running = cudaPathTracer.finishedSamplesPerPixel < cudaPathTracer.targetSamplesPerPixel;
            break;
        default:
            running = false;
            break;
        }
        if (!running)
        {
            stopwatch.stop();
            printStatistics();
        }
    }
    return running;
}
float ChromaRenderer::getProgress()
{
    if (state == State::RENDERING)
    {
        switch (rendererType)
        {
        case ChromaRenderer::RAYCAST:
            return (renderer.donePixelCount * invPixelCount);
            break;
        case ChromaRenderer::PATHTRACE:
            return (pathtracing.donePixelCount * invPixelCount);
            break;
        case ChromaRenderer::CUDAPATHTRACE:
            return cudaPathTracer.getProgress();
            break;
        default:
            return 1.0f;
            break;
        }
    }

    return 0.0f;
}
void ChromaRenderer::clearScene()
{
    if (!isIdle())
        return;
    scene.clear();
}

const std::string currentDateTime() // Get current date/time, format is YYYY-MM-DD.HH:mm:ss
{
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", &tstruct);

    return buf;
}
void ChromaRenderer::saveLog()
{
    /*std::ofstream file;
    std::string name("logs\\log-");
    name.append(currentDateTime());
    name.append(".txt");
    std::cout<<name<<std::endl;
    file.open(name.c_str());

    file.precision(6);
    file<<std::fixed;

    //write config
    //file<<"Width: "<<WIDTH<<std::endl;
    //file<<"Height: "<<HEIGHT<<std::endl;
    //file<<"Number of threads: "<<NUM_THREADS<<std::endl;
    //file<<"Pixels per tile: "<<PIXELSPERTILE<<std::endl;
    file<<"Scene triangle count: "<<scene.triangleCount()<<std::endl;
    #ifdef SUPERSAMPLING
    file<<"Supersampling: true"<<std::endl;
    #else
    file<<"Supersampling: false"<<std::endl;
    #endif
    #ifdef SHADOWRAY
    file<<"Shadow rays: true"<<std::endl;
    #else
    file<<"Shadow rays: false"<<std::endl;
    #endif
    #ifdef BOUNDINGBOXTEST
    file<<"Boundingbox: true"<<std::endl;
    #else
    file<<"Boundingbox: false"<<std::endl;
    #endif
    #ifdef SIMD
    file<<"SIMD: true"<<std::endl;
    #else
    file<<"SIMD: false"<<std::endl;
    #endif

    file<<std::endl<<"ElapsedTime(s)    MillionRay-TriangleIntesections/s"<<std::endl;

    if(renderer.log.size() > 0)
    {
        //long rayCount = HEIGHT*WIDTH;
        long rayCount = 0;
        //write the info
        for(int i = TEST_WARMUPS; i < renderer.log.size(); i++)
        {
            file<<renderer.log[i].elapsedTime
                <<"        "
                <<(rayCount*(renderer.log[i].triangleCount/renderer.log[i].elapsedTime)/1000000)
                <<std::endl;
        }
        //calc the average info
        double averageElapsedTime = 0.0;
        long averageTriangleCount = 0;
        for(int i = TEST_WARMUPS; i < renderer.log.size(); i++)
        {
            averageElapsedTime += renderer.log[i].elapsedTime;
            averageTriangleCount += renderer.log[i].triangleCount;
        }
        averageElapsedTime /= (renderer.log.size() - TEST_WARMUPS);
        averageTriangleCount /= (renderer.log.size() - TEST_WARMUPS);
        //write the average info
        file<<std::endl
            <<"Average"
            <<std::endl
            <<averageElapsedTime
            <<"        "
            <<(rayCount*(averageTriangleCount/averageElapsedTime)/1000000)
            <<std::endl;
    }
    file.close();*/
}

void ChromaRenderer::printStatistics()
{
    switch (rendererType)
    {
    case ChromaRenderer::RAYCAST: {
        std::cout << std::endl
                  << "Samples\\pixel:   " << settings.samplesperpixel << std::endl
                  << "Samples\\s:       "
                  << ((float)(settings.samplesperpixel * renderer.donePixelCount) /
                      (stopwatch.elapsedMillis.count() / 1000.0)) /
                         1000.0f
                  << "K" << std::endl
                  << "Rendering time:  " << stopwatch.elapsedMillis.count() / 1000.0 << "s" << std::endl
                  << std::endl;
    }
    break;
    case ChromaRenderer::PATHTRACE: {
        std::cout << std::endl
                  << "Samples\\pixel:   " << pathtracing.targetSamplesPerPixel << std::endl
                  << "Samples\\s:       "
                  << ((float)(pathtracing.targetSamplesPerPixel * pathtracing.donePixelCount) /
                      (stopwatch.elapsedMillis.count() / 1000.0)) /
                         1000.0f
                  << "K" << std::endl
                  << "Rendering time:  " << stopwatch.elapsedMillis.count() / 1000.0 << "s" << std::endl
                  << std::endl;
    }
    break;
    case ChromaRenderer::CUDAPATHTRACE:
        std::cout << std::endl
                  << "Samples\\pixel:   " << cudaPathTracer.finishedSamplesPerPixel << std::endl
                  << "Samples\\s:       "
                  << (((float)cudaPathTracer.finishedSamplesPerPixel * image.getWidth() * image.getHeight()) /
                      (stopwatch.elapsedMillis.count() / 1000.0)) /
                         1000.0f
                  << "K" << std::endl
                  << "Rendering time:  " << stopwatch.elapsedMillis.count() / 1000.0 << "s" << std::endl
                  << std::endl;
        break;
    }
}
