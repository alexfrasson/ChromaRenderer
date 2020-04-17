#include "ModelImporter.h"

#include <cmath>
#include <glm/glm.hpp>
#include <iostream>

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp> //OO version Header!
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>

// Example stream
class myStream : public Assimp::LogStream
{
  public:
    // Constructor
    myStream()
    {
        // empty
    }

    // Destructor
    ~myStream()
    {
        // empty
    }
    // Write womethink using your own functionality
    void write(const char* /*message*/)
    {
        // std::cout << message;
    }
};

const aiScene* importAssimpScene(std::string fileName)
{
    Assimp::DefaultLogger::create();
    // Select the kinds of messages you want to receive on this log stream
    const unsigned int severity =
        Assimp::Logger::Debugging | Assimp::Logger::Info | Assimp::Logger::Err | Assimp::Logger::Warn;
    // Attaching it to the default logger
    Assimp::DefaultLogger::get()->attachStream(new myStream(), severity);

    // Create an instance of the Importer class
    Assimp::Importer importer;
    importer.SetExtraVerbose(true);
    // Assimp scene object
    importer.ReadFile(fileName, aiProcess_Triangulate
                      //| aiProcess_JoinIdenticalVertices
                      //| aiProcess_SortByPType
                      //| aiProcess_PreTransformVertices
                      //| aiProcess_GenSmoothNormals		// Too slow
                      //| aiProcess_GenNormals
                      //| aiProcess_OptimizeMeshes
                      //| aiProcess_ImproveCacheLocality
    );
    const aiScene* assimpScene = importer.GetOrphanedScene();
    // If the import failed, report it
    if (!assimpScene)
    {
        std::cout << importer.GetErrorString() << std::endl;
        Assimp::DefaultLogger::kill();
        return NULL;
    }

    Assimp::DefaultLogger::kill();
    // We're done. Everything will be cleaned up by the importer destructor
    return assimpScene;
}

aiMatrix4x4 getLocalToWorldTransform(const aiNode* node)
{
    std::vector<aiMatrix4x4> transforms;

    while (node != nullptr)
    {
        transforms.push_back(node->mTransformation);
        node = node->mParent;
    }

    aiMatrix4x4 localToWorld = aiMatrix4x4();

    for (int i = (int)transforms.size() - 1; i >= 0; i--)
    // for (size_t i = 0; i < transforms.size(); i++)
    {
        localToWorld *= transforms[i];
    }

    return localToWorld;
}

void getTotalNumvberOfTrianglesAndVertices(const aiScene* scene,
                                           const aiNode* node,
                                           uint64_t* nTriangles,
                                           uint64_t* nVertices)
{
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        getTotalNumvberOfTrianglesAndVertices(scene, node->mChildren[i], nTriangles, nVertices);
    }

    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        *nTriangles += mesh->mNumFaces;
        *nVertices += mesh->mNumVertices;
    }
}

void printSceneInfo(const aiScene* scene)
{
    std::cout << "Scene info" << std::endl;
    std::cout << "\tMeshes:    " << scene->mNumMeshes << std::endl;

    uint64_t triangles = 0;
    uint64_t vertices = 0;
    // Count triangles and vertices
    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];

        triangles += mesh->mNumFaces;
        vertices += mesh->mNumVertices;
    }

    std::cout << "\tTriangles: " << triangles << std::endl;
    std::cout << "\tVertices:  " << vertices << std::endl << std::endl;

    triangles = 0;
    vertices = 0;

    getTotalNumvberOfTrianglesAndVertices(scene, scene->mRootNode, &triangles, &vertices);

    std::cout << "\tTriangles: " << triangles << std::endl;
    std::cout << "\tVertices:  " << vertices << std::endl << std::endl;

    std::cout << "\tCameras:   " << scene->mNumCameras << std::endl;
    std::cout << "\tLights:    " << scene->mNumLights << std::endl;
    std::cout << "\tMaterials: " << scene->mNumMaterials << std::endl;
    std::cout << "\tTextures:  " << scene->mNumTextures << std::endl << std::endl;
}
void printMeshInfo(const aiScene* scene)
{
    uint64_t triangles = 0;
    uint64_t vertices = 0;
    // Count triangles and vertices
    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];
        triangles += mesh->mNumFaces;
        vertices += mesh->mNumVertices;
    }

    std::cout << "Mesh info" << std::endl;
    std::cout << "\tTriangles: " << triangles << std::endl;
    std::cout << "\tVertices:  " << vertices << std::endl << std::endl;
}

// Compute the absolute transformation matrices of each node
void ComputeAbsoluteTransform(aiNode* pcNode)
{
    if (pcNode->mParent)
    {
        pcNode->mTransformation = pcNode->mParent->mTransformation * pcNode->mTransformation;
    }

    for (unsigned int i = 0; i < pcNode->mNumChildren; ++i)
    {
        ComputeAbsoluteTransform(pcNode->mChildren[i]);
    }
}

// If nodes have been transformed before hand
void convertToMeshRecursive(Scene& s, const aiScene* scene, const aiNode* node, Mesh* m, uint32_t& offset)
{
    aiMatrix4x4 mWorldIT = node->mTransformation;
    mWorldIT.Inverse().Transpose();
    aiMatrix3x3 m3x3 = aiMatrix3x3(mWorldIT);

    for (size_t i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // Copy vertex and normal data
        for (unsigned int k = 0; k < mesh->mNumVertices; k++)
        {
            aiVector3D v = node->mTransformation * mesh->mVertices[k];
            m->v.emplace_back(v.x, v.y, v.z);

            if (mesh->HasNormals())
            {
                aiVector3D n = (m3x3 * mesh->mNormals[k]).NormalizeSafe();
                m->n.emplace_back(n.x, n.y, n.z);

                // m->n.emplace_back(mesh->mNormals[k].x, mesh->mNormals[k].y, mesh->mNormals[k].z);
            }
        }

        // Copy everything
        for (unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            // Copy triangle data
            aiFace* face = &mesh->mFaces[j];
            if (face->mNumIndices != 3) // if the face is not a triangle
                continue;
            Triangle t;
            for (unsigned int k = 0; k < face->mNumIndices; k++)
            {
                t.v[k] = face->mIndices[k] + offset;
                if (mesh->HasNormals())
                    t.n[k] = face->mIndices[k] + offset;
            }

            t.vdata = &m->v;
            t.ndata = &m->n;

            t.precomputeStuff();

            // Material
            t.material = &s.materials[mesh->mMaterialIndex];

            m->t.emplace_back(t);
        }

        offset += mesh->mNumVertices;

        if (!mesh->HasNormals())
            m->genSmoothNormals();
    }

    for (size_t i = 0; i < node->mNumChildren; i++)
    {
        convertToMeshRecursive(s, scene, node->mChildren[i], m, offset);
    }
}
//
void convertToMeshRecursive(Scene& s,
                            const aiScene* scene,
                            const aiNode* node,
                            aiMatrix4x4 transform,
                            Mesh* m,
                            uint32_t& offset)
{
    transform = transform * node->mTransformation;

    aiMatrix4x4 mWorldIT = transform;
    mWorldIT.Inverse().Transpose();
    aiMatrix3x3 m3x3 = aiMatrix3x3(mWorldIT);

    /*aiQuaternion quat;
    aiVector3D scale;
    aiVector3D pos;
    transform.Decompose(scale, quat, pos);*/

    for (size_t i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // Copy vertex and normal data
        for (unsigned int k = 0; k < mesh->mNumVertices; k++)
        {
            aiVector3D v = transform * mesh->mVertices[k];
            m->v.emplace_back(v.x, v.y, v.z);

            if (mesh->HasNormals())
            {
                // aiVector3D n = quat.Rotate(mesh->mNormals[k]).NormalizeSafe();
                aiVector3D n = (m3x3 * mesh->mNormals[k]).NormalizeSafe();
                m->n.emplace_back(n.x, n.y, n.z);

                // m->n.emplace_back(mesh->mNormals[k].x, mesh->mNormals[k].y, mesh->mNormals[k].z);
            }
        }

        // Copy everything
        for (unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            // Copy triangle data
            aiFace* face = &mesh->mFaces[j];
            if (face->mNumIndices != 3) // if the face is not a triangle
                continue;
            Triangle t;
            for (unsigned int k = 0; k < face->mNumIndices; k++)
            {
                t.v[k] = face->mIndices[k] + offset;
                if (mesh->HasNormals())
                    t.n[k] = face->mIndices[k] + offset;
            }

            t.vdata = &m->v;
            t.ndata = &m->n;

            t.precomputeStuff();

            // Material
            t.material = &s.materials[mesh->mMaterialIndex];

            m->t.emplace_back(t);
        }

        offset += mesh->mNumVertices;

        if (!mesh->HasNormals())
            m->genSmoothNormals();
    }

    for (size_t i = 0; i < node->mNumChildren; i++)
    {
        convertToMeshRecursive(s, scene, node->mChildren[i], transform, m, offset);
    }
}

bool convert(const aiScene* scene, Mesh& m)
{
    int triangles = 0;
    int vertices = 0;
    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];
        triangles += mesh->mNumFaces;
        vertices += mesh->mNumVertices;
    }

    // Reserve memory
    m.t.reserve(triangles);
    m.v.reserve(vertices);
    m.n.reserve(vertices);

    uint32_t offset = 0;

    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];

        // Reserve memory
        // m.t.reserve(mesh->mNumFaces);
        // m.v.reserve(mesh->mNumVertices);
        // m.n.reserve(mesh->mNumVertices);

        // Copy vertex and normal data
        for (unsigned int k = 0; k < mesh->mNumVertices; k++)
        {
            m.v.emplace_back(mesh->mVertices[k].x, mesh->mVertices[k].y, mesh->mVertices[k].z);
            if (mesh->HasNormals())
                m.n.emplace_back(mesh->mNormals[k].x, mesh->mNormals[k].y, mesh->mNormals[k].z);
        }

        // Copy everything
        for (unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            // Copy triangle data
            aiFace* face = &mesh->mFaces[j];
            if (face->mNumIndices != 3) // if the face is not a triangle
                continue;
            Triangle t;
            for (unsigned int k = 0; k < face->mNumIndices; k++)
            {
                t.v[k] = face->mIndices[k] + offset;
                if (mesh->HasNormals())
                    t.n[k] = face->mIndices[k] + offset;
            }

            t.vdata = &m.v;
            t.ndata = &m.n;

            t.precomputeStuff();

            m.t.emplace_back(t);
        }

        offset += mesh->mNumVertices;

        if (!mesh->HasNormals())
            m.genSmoothNormals();
    }

    m.genBoundingBox();

    return true;
}
Mesh* convertToMesh(const aiScene* scene, Scene& s)
{
    uint64_t triangles = 0;
    uint64_t vertices = 0;

    getTotalNumvberOfTrianglesAndVertices(scene, scene->mRootNode, &triangles, &vertices);

    Mesh* m = new Mesh();
    // Reserve memory
    m->t.reserve(triangles);
    m->v.reserve(vertices);
    m->n.reserve(vertices);

    uint32_t offset = 0;

    // ComputeAbsoluteTransform(scene->mRootNode);

    convertToMeshRecursive(s, scene, scene->mRootNode, aiMatrix4x4(), m, offset);

    m->genBoundingBox();

    return m;
}
void printMaterialInfo(const aiMaterial* mat)
{
    // aiColor3D color;
    // aiString str;
    // Name
    // if (AI_SUCCESS == mat->Get(AI_MATKEY_NAME, str))
    //	std::cout << "Name: " << str.C_Str() << std::endl;

    //// Diffuse color
    // if (AI_SUCCESS == mat->Get(AI_MATKEY_COLOR_DIFFUSE, color))
    //	std::cout << "Diffuse color: (" << color.r << ", " << color.g << ", " << color.b << ")" << std::endl;

    //// Emissive color
    // if (AI_SUCCESS == mat->Get(AI_MATKEY_COLOR_EMISSIVE, color))
    //	std::cout << "Emissive color: (" << color.r << ", " << color.g << ", " << color.b << ")" << std::endl;

    for (uint32_t j = 0; j < mat->mNumProperties; j++)
    {
        aiMaterialProperty* prop = mat->mProperties[j];

        std::cout << prop->mKey.C_Str() << ": ";

        aiPropertyTypeInfo type = prop->mType;

        switch (type)
        {
        case aiPTI_Float: {
            float* farr = (float*)prop->mData;
            for (size_t i = 0; i < prop->mDataLength / sizeof(float); i++)
                std::cout << farr[i] << ", ";
            std::cout << std::endl;
            break;
        }
        case aiPTI_Double: {
            double* darr = (double*)prop->mData;
            for (size_t i = 0; i < prop->mDataLength / sizeof(double); i++)
                std::cout << darr[i] << ", ";
            std::cout << std::endl;
            break;
        }
        case aiPTI_String: {
            // Extracted from assimp source:
            // ai_assert(prop->mDataLength >= 5);

            //// The string is stored as 32 but length prefix followed by zero-terminated UTF8 data
            // pOut->length = static_cast<unsigned int>(*reinterpret_cast<uint32_t*>(prop->mData));

            // ai_assert(pOut->length + 1 + 4 == prop->mDataLength);
            // ai_assert(!prop->mData[prop->mDataLength - 1]);
            // memcpy(pOut->data, prop->mData + 4, pOut->length + 1);

            for (size_t i = 4; i < prop->mDataLength; i++)
                std::cout << prop->mData[i];
            std::cout << std::endl;
            break;
        }
        case aiPTI_Integer: {
            int* iarr = (int*)prop->mData;
            for (size_t i = 0; i < prop->mDataLength / sizeof(int); i++)
                std::cout << iarr[i] << ", ";
            std::cout << std::endl;
            break;
        }
        case aiPTI_Buffer:
            break;
        default:
            break;
        }
    }

    std::cout << std::endl;
}
void extractMaterials(const aiScene* aiscene, Scene& s)
{
    std::cout << std::endl << "Extracting materials... " << std::endl;

    if (!aiscene->HasMaterials())
    {
        Material m;
        m.kd = Color::GREEN;
        m.ke = Color::BLACK;
        s.materials.emplace_back(m);
        return;
    }

    s.materials.reserve(aiscene->mNumMaterials);

    for (uint32_t i = 0; i < aiscene->mNumMaterials; i++)
    {
        aiMaterial* mat = aiscene->mMaterials[i];

        printMaterialInfo(mat);

        Material m = Material();
        aiColor3D color;
        aiString str;

        // Get diffuse
        mat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        m.kd.r = color.r;
        m.kd.g = color.g;
        m.kd.b = color.b;

        // Get emissive
        if (AI_SUCCESS == mat->Get(AI_MATKEY_COLOR_EMISSIVE, color))
        {
            m.ke.r = color.r;
            m.ke.g = color.g;
            m.ke.b = color.b;
        }

        // Get transparency
        if (AI_SUCCESS == mat->Get(AI_MATKEY_COLOR_TRANSPARENT, color))
        {
            m.transparent.r = color.r;
            m.transparent.g = color.g;
            m.transparent.b = color.b;
        }

        // Get name
        if (AI_SUCCESS == mat->Get(AI_MATKEY_NAME, str))
            m.name = std::string(str.C_Str());

        s.materials.emplace_back(m);
    }
}

bool convert(const aiScene* aiscene, Scene& s)
{
    s.clear();

    extractMaterials(aiscene, s);

    s.meshes.emplace_back(convertToMesh(aiscene, s));

    if (aiscene->HasCameras())
    {
        const aiCamera* cam = aiscene->mCameras[0];

        const aiNode* nd = aiscene->mRootNode->FindNode(cam->mName);

        aiMatrix4x4 nodeTransform = getLocalToWorldTransform(nd);

        // multiply all properties of the camera with the absolute
        // transformation of the corresponding node
        /*aiVector3D camPos = nodeTransform * cam->mPosition;
        aiVector3D camLookAt = aiMatrix3x3(nodeTransform) * cam->mLookAt;
        aiVector3D camUp = aiMatrix3x3(nodeTransform) * cam->mUp;*/

        aiMatrix4x4 mWorldIT = nodeTransform;
        mWorldIT.Inverse().Transpose();
        aiMatrix3x3 m3x3 = aiMatrix3x3(mWorldIT);

        // multiply all properties of the camera with the absolute
        // transformation of the corresponding node
        aiVector3D camPos = nodeTransform * cam->mPosition;
        aiVector3D camLookAt = m3x3 * cam->mLookAt;
        aiVector3D camUp = m3x3 * cam->mUp;

        // aiMatrix4x4 mat;

        // aiNode* camNode = aiscene->mRootNode->FindNode(cam->mName);

        ////aiMatrix4x4 nodeTransform = camNode->mTransformation;
        // aiMatrix4x4 nodeTransform = getLocalToWorldTransform(camNode);

        // aiVector3D scale, position;
        // aiQuaternion quaternion;

        // nodeTransform.Decompose(scale, quaternion, position);

        // aiVector3D camPos = nodeTransform * cam->mPosition;
        // aiVector3D camUp = quaternion.Rotate(cam->mUp);
        // aiVector3D camLookAt = quaternion.Rotate(cam->mLookAt);

        /*aiVector3D camPos = cam->mPosition;
        aiVector3D camLookAt = cam->mLookAt;
        aiVector3D camUp = cam->mUp;*/

        s.camera.eye = glm::vec3(camPos.x, camPos.y, camPos.z);
        s.camera.forward = glm::normalize(glm::vec3(camLookAt.x, camLookAt.y, camLookAt.z));
        s.camera.up = glm::normalize(glm::vec3(camUp.x, camUp.y, camUp.z));
        s.camera.right = glm::normalize(glm::cross(s.camera.forward, s.camera.up));

        // tan(FOV/2) = (screenSize/2) / screenPlaneDistance
        // tan(FOV_H/2) = (screen_width/2) / screenPlaneDistance
        // tan(FOV_V / 2) = (screen_height / 2) / screenPlaneDistance
        // tan(FOV_H/2) / screen_width = tan(FOV_V/2) / screen_height
        s.camera.aspectRatio = cam->mAspect;
        s.camera.width = 640;
        s.camera.d = ((float)s.camera.width / 2.0f) / tan(cam->mHorizontalFOV / 2.0f);
        s.camera.height = static_cast<int>(s.camera.width / cam->mAspect);
        s.camera.horizontalFOV(cam->mHorizontalFOV);

        // s.camera.computeUVW();
    }
    else
    {
        s.camera.fit(s.getBoundingBox());
    }

    return true;
}

void ModelImporter::importcbm(std::string fileName, std::function<void(Mesh*)> cb)
{
    Mesh* m = new Mesh();
    import(fileName, *m);
    cb(m);
}
bool ModelImporter::import(std::string fileName, Mesh& m)
{
    std::cout << "----------------------------<Importer>---------------------------" << std::endl;
    std::cout << "Loading mesh..." << std::endl;
    const aiScene* assimpScene = importAssimpScene(fileName);
    if (!assimpScene)
    {
        std::cout << "Assimp failed to load the file!" << std::endl;
        std::cout << "---------------------------</Importer>---------------------------" << std::endl;
        return false;
    }
    std::cout << "Done!" << std::endl << std::endl;

    printMeshInfo(assimpScene);

    std::cout << "Converting mesh..." << std::endl;
    convert(assimpScene, m);
    std::cout << "Done!" << std::endl << std::endl;

    delete assimpScene;

    std::cout << "Import of mesh " << fileName.c_str() << " succeeded." << std::endl;
    std::cout << "---------------------------</Importer>---------------------------" << std::endl;
    return true;
}

void ModelImporter::importcbscene(std::string fileName, Scene& s, std::function<void()> cb)
{
    import(fileName, s);
    cb();
}
bool ModelImporter::import(std::string fileName, Scene& s)
{
    std::cout << "----------------------------<Importer>---------------------------" << std::endl;
    std::cout << "Assimp " << aiGetVersionMajor() << "." << aiGetVersionMinor() << std::endl;
    std::cout << "Loading scene..." << std::endl;
    const aiScene* assimpScene = importAssimpScene(fileName);
    if (!assimpScene)
    {
        std::cout << "Assimp failed to load the file!" << std::endl;
        std::cout << "---------------------------</Importer>---------------------------" << std::endl;
        return false;
    }
    std::cout << "Done!" << std::endl << std::endl;

    printSceneInfo(assimpScene);

    std::cout << "Converting scene..." << std::endl;
    convert(assimpScene, s);
    std::cout << "Done!" << std::endl << std::endl;

    delete assimpScene;

    std::cout << "Import of scene " << fileName.c_str() << " succeeded." << std::endl;
    std::cout << "---------------------------</Importer>---------------------------" << std::endl;
    return true;
}