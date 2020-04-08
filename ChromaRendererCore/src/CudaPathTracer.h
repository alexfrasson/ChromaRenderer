#pragma once

#include <Camera.h>
#include <Image.h>
#include <Scene.h>
#include <CudaPathTracerKernel.h>
#include <RendererSettings.h>
#include <Stopwatch.h>
#include <GlslProgram.h>

class CudaPathTracer
{
public:
	int iteration = 0;

	unsigned int targetSamplesPerPixel;
	unsigned int finishedSamplesPerPixel;

	CudaPathTracer();
	~CudaPathTracer();
	void init(Scene& scene);
	void init(float* hdriEnvData, int hdriEnvWidth, int hdriEnvHeight);
	void init(Image& img, Camera& cam);
	void uploadMaterials(Scene& scene);
	void renderThread(bool& abort);
	void render();
	void setSettings(RendererSettings& settings);
	void copyFrameToTexture();
	void dispatchComputeShader(bool sync);
	float getProgress();
	float instantRaysPerSec();

	float gammaCorrectionScale = 1.0f;

private:
	struct RegisteredImage
	{
		int width = -1, height = -1;
		GLuint texID = 0;
		cudaGraphicsResource* cudaTextureResource = nullptr;

		bool changed(const Image& img);
	};

	RegisteredImage registeredImage;

	cudaStream_t stream;
	cudaStream_t cpystream;

	CudaLinearBvhNode *dev_cudaLinearBVHNodes = 0;
	unsigned int nCudaLinearBVHNodes = 0;

	CudaTriangle *dev_cudaTrianglesBVH = 0;
	unsigned int nCudaTrianglesBVH = 0;

	CudaMaterial *dev_cudaMaterials = 0;
	unsigned int nCudaMaterials = 0;

	CudaCamera cuda_cam;
	CudaEnviromentSettings enviromentSettings;

	cudaArray* envArray = 0;

	float4* dev_accuBuffer = 0;
	CudaPathIteration* dev_pathIterationBuffer = 0;

	Stopwatch stopwatch;
	std::chrono::milliseconds lastIterationElapsedMillis;

	GLSLProgram* computeShader;
};