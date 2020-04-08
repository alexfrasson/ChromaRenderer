#include <CudaPathTracer.h>
#include <CudaPathTracerKernel.h>

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>
#include <thread>
#include <chrono>
#include <iostream>
#include <cmath>


#define SAFE_CUDA_FREE(x) if (x) cudaErrorCheck(cudaFree(x));
#define SAFE_CUDA_FREE_ARRAY(x) if (x) cudaErrorCheck(cudaFreeArray(x));

using namespace std;

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n", (int)devProp.major);
	printf("Minor revision number:         %d\n", (int)devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %d\n", (int)devProp.totalGlobalMem);
	printf("Total shared memory per block: %d\n", (int)devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", (int)devProp.regsPerBlock);
	printf("Warp size:                     %d\n", (int)devProp.warpSize);
	printf("Maximum memory pitch:          %d\n", (int)devProp.memPitch);
	printf("Maximum threads per block:     %d\n", (int)devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, (int)devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, (int)devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", (int)devProp.clockRate);
	printf("Total constant memory:         %d\n", (int)devProp.totalConstMem);
	printf("Texture alignment:             %d\n", (int)devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", (int)devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

// Converts Camera to CudaCamera
CudaCamera CameraToCudaCamera(Camera cam)
{
	CudaCamera cudaCam;
	cudaCam.width = cam.width;
	cudaCam.height = cam.height;
	cudaCam.d = cam.d;
	cudaCam.right.x = cam.right.x; cudaCam.right.y = cam.right.y; cudaCam.right.z = cam.right.z;
	cudaCam.up.x = cam.up.x; cudaCam.up.y = cam.up.y; cudaCam.up.z = cam.up.z;
	cudaCam.forward.x = cam.forward.x; cudaCam.forward.y = cam.forward.y; cudaCam.forward.z = cam.forward.z;
	cudaCam.eye.x = cam.eye.x; cudaCam.eye.y = cam.eye.y; cudaCam.eye.z = cam.eye.z;
	return cudaCam;
}
// Merge every mesh into one single CudaTriangle array
vector<CudaTriangle> SceneToCudaTriangles(Scene& scene)
{
	vector<CudaTriangle> cudaTriangles;
	cudaTriangles.reserve(scene.triangleCount());

	for each (Mesh* m in scene.meshes)
	{
		for each (Triangle t in m->t)
		{
			CudaTriangle ct;

			// Copy vertices and normals
			for (int i = 0; i < 3; i++)
			{
				ct.v[i].x = t.getVertex(i)->x;
				ct.v[i].y = t.getVertex(i)->y;
				ct.v[i].z = t.getVertex(i)->z;

				ct.n[i].x = t.getNormal(i)->x;
				ct.n[i].y = t.getNormal(i)->y;
				ct.n[i].z = t.getNormal(i)->z;
			}

			// Find material index
			for (int i = 0; i < scene.materials.size(); i++)
			{
				if (t.material == &scene.materials[i])
				{
					ct.material = i;
					break;
				}
			}

			cudaTriangles.push_back(ct);
		}
	}

	return cudaTriangles;
}
vector<CudaMaterial> SceneToCudaMaterials(Scene& scene)
{
	vector<CudaMaterial> cudaMaterials;

	for each (Material m in scene.materials)
	{
		CudaMaterial cm;
		cm.kd.x = m.kd.r; cm.kd.y = m.kd.g; cm.kd.z = m.kd.b;
		cm.ke.x = m.ke.r; cm.ke.y = m.ke.g; cm.ke.z = m.ke.b;
		cm.transparent.x = m.transparent.r; cm.transparent.y = m.transparent.g; cm.transparent.z = m.transparent.b;
		cudaMaterials.push_back(cm);
	}

	return cudaMaterials;
}
vector<CudaLinearBvhNode> SceneToCudaLinearBvhNode(Scene& scene)
{
	// Lets assume this is a bvh :)
	BVH* bvh = (BVH*)scene.sps;

	vector<CudaLinearBvhNode> cudaLinearBVH;
	cudaLinearBVH.reserve(bvh->nNodes);

	for (unsigned int i = 0; i < bvh->nNodes; i++)
	{
		CudaLinearBvhNode n;
		n.axis = bvh->lroot[i].axis;
		n.nPrimitives = bvh->lroot[i].nPrimitives;
		n.primitivesOffset = bvh->lroot[i].primitivesOffset;
		n.bbox.max.x = bvh->lroot[i].bbox.max.x; n.bbox.max.y = bvh->lroot[i].bbox.max.y; n.bbox.max.z = bvh->lroot[i].bbox.max.z;
		n.bbox.min.x = bvh->lroot[i].bbox.min.x; n.bbox.min.y = bvh->lroot[i].bbox.min.y; n.bbox.min.z = bvh->lroot[i].bbox.min.z;

		cudaLinearBVH.push_back(n);
	}

	return cudaLinearBVH;
}
vector<CudaTriangle> SceneToCudaTrianglesBVH(Scene& scene)
{
	// Lets assume this is a bvh :)
	BVH* bvh = (BVH*)scene.sps;

	vector<CudaTriangle> cudaTriangles;
	cudaTriangles.reserve(bvh->triangles.size());

	for each (Triangle* t in bvh->triangles)
	{
		CudaTriangle ct;

		// Copy vertices and normals
		for (int i = 0; i < 3; i++)
		{
			ct.v[i].x = t->getVertex(i)->x;
			ct.v[i].y = t->getVertex(i)->y;
			ct.v[i].z = t->getVertex(i)->z;

			ct.n[i].x = t->getNormal(i)->x;
			ct.n[i].y = t->getNormal(i)->y;
			ct.n[i].z = t->getNormal(i)->z;
		}

		// Find material index
		for (int i = 0; i < scene.materials.size(); i++)
		{
			if (t->material == &scene.materials[i])
			{
				ct.material = i;
				break;
			}
		}

		cudaTriangles.push_back(ct);
	}

	return cudaTriangles;
}

// this hash function calculates a new random number generator seed for each frame, based on framenumber  
unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}


CudaPathTracer::CudaPathTracer()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaErrorCheck(cudaSetDevice(0));
	cudaErrorCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	cudaErrorCheck(cudaStreamCreateWithFlags(&cpystream, cudaStreamNonBlocking));

	// Number of CUDA devices
	int devCount;
	cudaErrorCheck(cudaGetDeviceCount(&devCount));
	cout << "CUDA Device Query..." << endl;
	cout << "There are " << devCount << " CUDA devices." << endl;

	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		cout << endl << "CUDA Device " << i << endl;
		cudaDeviceProp devProp;
		cudaErrorCheck(cudaGetDeviceProperties(&devProp, i));
		printDevProp(devProp);
		cout << endl << endl;
	}

	//try
	{
		computeShader = new GLSLProgram();
		computeShader->compileShader("./ChromaRendererCore/shaders/convergence.glsl", GLSLShader::COMPUTE);
		computeShader->link();
		computeShader->validate();
		//computeShader->printActiveAttribs();
	}
	//catch (GLSLProgramException &e)
	//{
	//	std::cerr << e.what() << std::endl;
	//}
}

CudaPathTracer::~CudaPathTracer()
{
	cudaErrorCheck(cudaDeviceSynchronize());

	if (registeredImage.cudaTextureResource != nullptr)
		cudaErrorCheck(cudaGraphicsUnregisterResource(registeredImage.cudaTextureResource));

	SAFE_CUDA_FREE(dev_cudaTrianglesBVH)
	SAFE_CUDA_FREE(dev_cudaLinearBVHNodes)
	SAFE_CUDA_FREE(dev_cudaMaterials)
	SAFE_CUDA_FREE(dev_accuBuffer)
	SAFE_CUDA_FREE(dev_pathIterationBuffer)

	cudaErrorCheck(cudaDestroyTextureObject(enviromentSettings.texObj));

	SAFE_CUDA_FREE_ARRAY(envArray)

	cudaErrorCheck(cudaStreamDestroy(stream));
	cudaErrorCheck(cudaStreamDestroy(cpystream));

	cudaErrorCheck(cudaDeviceReset());
}

void CudaPathTracer::copyFrameToTexture()
{
	cudaArray *aarray;

	cudaErrorCheck(cudaGraphicsMapResources(1, &registeredImage.cudaTextureResource, cpystream));
	cudaErrorCheck(cudaGraphicsSubResourceGetMappedArray(&aarray, registeredImage.cudaTextureResource, 0, 0));

	cudaErrorCheck(cudaMemcpyToArrayAsync(aarray, 0, 0, dev_accuBuffer, registeredImage.width * registeredImage.height * sizeof(float4), cudaMemcpyDeviceToDevice, cpystream));

	cudaErrorCheck(cudaGraphicsUnmapResources(1, &registeredImage.cudaTextureResource, cpystream));
}

void CudaPathTracer::render()
{
	bool firstIteration = iteration == 0;

	{
		cudaErrorCheck(cudaGetLastError());

		if (firstIteration)
		{
			cudaErrorCheck(cudaStreamSynchronize(stream));
		}
		else
		{
			cudaError err = cudaStreamQuery(stream);

			if (err == cudaErrorNotReady)
				return;
			else if (err == cudaSuccess)
			{
				stopwatch.stop();
				lastIterationElapsedMillis = stopwatch.elapsedMillis;
				stopwatch.start();
				copyFrameToTexture();
				dispatchComputeShader(false);
			}
			else
				cudaErrorCheck(err);
		}
	}

	{
		int iteraionsThisFrame = firstIteration ? MAX_PATH_DEPTH : 1;

		//std::cout << "Tracing: iteraionsThisFrame(" << iteraionsThisFrame << ")" << std::endl;

		for (int i = 0; i < iteraionsThisFrame; i++)
		{
			// calculate a new seed for the random number generator, based on the framenumber
			unsigned int hashedframes = WangHash(iteration);

			trace(stream, dev_pathIterationBuffer, dev_accuBuffer, dim3(registeredImage.width, registeredImage.height), cuda_cam, dev_cudaTrianglesBVH, nCudaTrianglesBVH, dev_cudaMaterials, nCudaMaterials, hashedframes, dev_cudaLinearBVHNodes, enviromentSettings);

			cudaErrorCheck(cudaGetLastError());

			if (iteration % MAX_PATH_DEPTH == 0)
				finishedSamplesPerPixel++;

			iteration++;
		}

		if (firstIteration)
		{
			//std::cout << "First iteration. Synchronizing." << std::endl;
			cudaErrorCheck(cudaStreamSynchronize(stream));

			copyFrameToTexture();
			dispatchComputeShader(true);
		}
	}
}

void CudaPathTracer::renderThread(bool& abort)
{
	finishedSamplesPerPixel = 0;

	for (size_t i = 0; i < targetSamplesPerPixel * MAX_PATH_DEPTH; i++)
	{
		// calculate a new seed for the random number generator, based on the framenumber
		unsigned int hashedframes = WangHash(i);

		trace(stream, dev_pathIterationBuffer, dev_accuBuffer, dim3(registeredImage.width, registeredImage.height), cuda_cam, dev_cudaTrianglesBVH, nCudaTrianglesBVH, dev_cudaMaterials, nCudaMaterials, hashedframes, dev_cudaLinearBVHNodes, enviromentSettings);

		cudaErrorCheck(cudaGetLastError());
		cudaErrorCheck(cudaStreamSynchronize(stream));

		if (i % MAX_PATH_DEPTH == 0)
			finishedSamplesPerPixel++;

		if (abort)
			break;
	}
}

void CudaPathTracer::init(float* hdriEnvData, int hdriEnvWidth, int hdriEnvHeight)
{
	//Load reference image from image (output)
	unsigned int size = hdriEnvWidth * hdriEnvHeight * 3 * sizeof(float);

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	cudaErrorCheck(cudaMallocArray(&envArray, &channelDesc, hdriEnvWidth, hdriEnvHeight));

	// Copy to device memory some data located at address h_data
	// in host memory 
	cudaErrorCheck(cudaMemcpyToArray(envArray, 0, 0, hdriEnvData, size, cudaMemcpyHostToDevice));

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = envArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaErrorCheck(cudaCreateTextureObject(&enviromentSettings.texObj, &resDesc, &texDesc, NULL));
}

void CudaPathTracer::init(Scene& scene)
{
	cout << endl << "CUDA PATH TRACER" << endl;
	
	size_t free, total;

	cudaErrorCheck(cudaMemGetInfo(&free, &total));
	cout << "Free memory: " << free / (1024*1024) << "MB" << endl;
	cout << "Total memory: " << total / (1024 * 1024) << "MB" << endl;

	// <Convert scene>
	/*vector<CudaTriangle> cudaTriangles = SceneToCudaTriangles(scene);
	nCudaTriangles = cudaTriangles.size();
	cout << "Triangles: " << cudaTriangles.size() << endl;
	cout << "Triangles size: " << (cudaTriangles.size() * sizeof(CudaTriangle)) / (1024) << "KB" << endl;
	cudaErrorCheck(cudaMalloc((void**)&dev_cudaTriangles, cudaTriangles.size() * sizeof(CudaTriangle)));
	cudaErrorCheck(cudaMemcpy(dev_cudaTriangles, &cudaTriangles[0], cudaTriangles.size() * sizeof(CudaTriangle), cudaMemcpyHostToDevice));*/

	vector<CudaTriangle> cudaTrianglesBVH = SceneToCudaTrianglesBVH(scene);
	nCudaTrianglesBVH = cudaTrianglesBVH.size();
	cout << "Triangles BVH: " << cudaTrianglesBVH.size() << endl;
	cout << "Triangles BVH size: " << (cudaTrianglesBVH.size() * sizeof(CudaTriangle)) / (1024) << "KB" << endl;
	cudaErrorCheck(cudaMalloc((void**)&dev_cudaTrianglesBVH, cudaTrianglesBVH.size() * sizeof(CudaTriangle)));
	//cudaErrorCheck(cudaMemcpy(dev_cudaTrianglesBVH, &cudaTrianglesBVH[0], cudaTrianglesBVH.size() * sizeof(CudaTriangle), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpyAsync(dev_cudaTrianglesBVH, &cudaTrianglesBVH[0], cudaTrianglesBVH.size() * sizeof(CudaTriangle), cudaMemcpyHostToDevice, stream));

	vector<CudaLinearBvhNode> cudaLinearBVH = SceneToCudaLinearBvhNode(scene);
	nCudaLinearBVHNodes = cudaLinearBVH.size();
	cout << "CudaLinearBVHNodes: " << cudaLinearBVH.size() << endl;
	cout << "CudaLinearBVHNodes size: " << (cudaLinearBVH.size() * sizeof(CudaLinearBvhNode)) / (1024) << "KB" << endl;
	cudaErrorCheck(cudaMalloc((void**)&dev_cudaLinearBVHNodes, cudaLinearBVH.size() * sizeof(CudaLinearBvhNode)));
	//cudaErrorCheck(cudaMemcpy(dev_cudaLinearBVHNodes, &cudaLinearBVH[0], cudaLinearBVH.size() * sizeof(CudaLinearBvhNode), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpyAsync(dev_cudaLinearBVHNodes, &cudaLinearBVH[0], cudaLinearBVH.size() * sizeof(CudaLinearBvhNode), cudaMemcpyHostToDevice, stream));

	vector<CudaMaterial> cudaMaterials = SceneToCudaMaterials(scene);
	nCudaMaterials = cudaMaterials.size();
	cout << "Materials: " << cudaMaterials.size() << endl;
	cout << "Materials size: " << (cudaMaterials.size() * sizeof(CudaMaterial)) / (1024) << "KB" << endl;
	cudaErrorCheck(cudaMalloc((void**)&dev_cudaMaterials, cudaMaterials.size() * sizeof(CudaMaterial)));
	//cudaErrorCheck(cudaMemcpy(dev_cudaMaterials, &cudaMaterials[0], cudaMaterials.size() * sizeof(CudaMaterial), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpyAsync(dev_cudaMaterials, &cudaMaterials[0], cudaMaterials.size() * sizeof(CudaMaterial), cudaMemcpyHostToDevice, stream));

	cudaErrorCheck(cudaGetLastError());
	cudaErrorCheck(cudaStreamSynchronize(stream));
}

void CudaPathTracer::init(Image& img, Camera& cam)
{
	assert(img.textureID > 0);
	assert(img.getWidth() > 0 && img.getHeight() > 0);

	if (registeredImage.changed(img))
	{
		cudaErrorCheck(cudaFree(dev_accuBuffer));
		cudaErrorCheck(cudaMalloc((void**)&dev_accuBuffer, img.getWidth() * img.getHeight() * sizeof(float4)));

		assert(dev_accuBuffer != nullptr);

		cudaErrorCheck(cudaFree(dev_pathIterationBuffer));
		cudaErrorCheck(cudaMalloc((void**)&dev_pathIterationBuffer, img.getWidth() * img.getHeight() * sizeof(CudaPathIteration)));

		cudaErrorCheck(cudaDeviceSynchronize());

		assert(dev_pathIterationBuffer != nullptr);

		if (registeredImage.cudaTextureResource != nullptr)
			cudaErrorCheck(cudaGraphicsUnregisterResource(registeredImage.cudaTextureResource));

		// Only call Cuda/OpenGL interop stuff from within the OpenGL context thread!
		cudaErrorCheck(cudaGraphicsGLRegisterImage(&registeredImage.cudaTextureResource, img.textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

		registeredImage.width = img.getWidth();
		registeredImage.height = img.getHeight();
		registeredImage.texID = img.textureID;
	}

	cudaErrorCheck(cudaMemsetAsync(dev_accuBuffer, 0, registeredImage.width * registeredImage.height * sizeof(float4), stream));
	cudaErrorCheck(cudaMemsetAsync(dev_pathIterationBuffer, 0, registeredImage.width * registeredImage.height * sizeof(CudaPathIteration), stream));

	cudaErrorCheck(cudaGetLastError());
	cudaErrorCheck(cudaStreamSynchronize(stream));

	cuda_cam = CameraToCudaCamera(cam);

	finishedSamplesPerPixel = 0;
	iteration = 0;
}

void CudaPathTracer::uploadMaterials(Scene& scene)
{
	vector<CudaMaterial> cudaMaterials = SceneToCudaMaterials(scene);
	nCudaMaterials = cudaMaterials.size();
	if (dev_cudaMaterials == nullptr)
		cudaErrorCheck(cudaMalloc((void**)&dev_cudaMaterials, cudaMaterials.size() * sizeof(CudaMaterial)));
	//cudaErrorCheck(cudaMemcpy(dev_cudaMaterials, &cudaMaterials[0], cudaMaterials.size() * sizeof(CudaMaterial), cudaMemcpyHostToDevice));
	cudaErrorCheck(cudaMemcpyAsync(dev_cudaMaterials, &cudaMaterials[0], cudaMaterials.size() * sizeof(CudaMaterial), cudaMemcpyHostToDevice, stream));

	cudaErrorCheck(cudaGetLastError());
	cudaErrorCheck(cudaStreamSynchronize(stream));
}

void CudaPathTracer::setSettings(RendererSettings& settings)
{
	targetSamplesPerPixel = settings.samplesperpixel;
	enviromentSettings.enviromentLightColor = make_float3(settings.enviromentLightColor.x, settings.enviromentLightColor.y, settings.enviromentLightColor.z);
	enviromentSettings.enviromentLightIntensity = settings.enviromentLightIntensity;
	gammaCorrectionScale = settings.enviromentLightIntensity;
}

float CudaPathTracer::getProgress()
{
	return ((float)finishedSamplesPerPixel / (float)targetSamplesPerPixel);
}

float CudaPathTracer::instantRaysPerSec()
{
	return (registeredImage.width * (double)registeredImage.height) / (lastIterationElapsedMillis.count() * 0.001);
}

void CudaPathTracer::dispatchComputeShader(bool sync)
{
	computeShader->use();
	computeShader->setUniform("enviromentLightIntensity", gammaCorrectionScale);
	computeShader->setUniform("imgSnapshot", 0);
	//shader.setUniform("lastRenderedBuffer", readFromFboTex0 ? 2 : 4);

	glBindImageTexture(0, registeredImage.texID, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	//glBindImageTexture(2, aaFbo0.colorTexId, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
	//glBindImageTexture(4, aaFbo1.colorTexId, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

	int nGroupsX = static_cast<int>(ceilf(registeredImage.width / 16.0f));
	int nGroupsY = static_cast<int>(ceilf(registeredImage.height / 16.0f));

	//if (showHeatmap)
	//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, convergedPixelCountBufferId);

	glDispatchCompute(nGroupsX, nGroupsY, 1);
	//if (showHeatmap)
	//	glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);

	if (sync)
	{
		glMemoryBarrier(GL_ALL_BARRIER_BITS);
		//glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		//glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);
	}

	//if (showHeatmap)
	//{
	//	result = *(GLuint*)glMapBuffer(GL_ATOMIC_COUNTER_BUFFER, GL_READ_ONLY);
	//	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	//	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
	//}
}

bool CudaPathTracer::RegisteredImage::changed(const Image& img)
{
	return img.getWidth() != width || img.getHeight() != height || img.textureID != texID;
}
