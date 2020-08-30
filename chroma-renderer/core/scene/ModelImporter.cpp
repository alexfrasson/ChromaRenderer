#include "chroma-renderer/core/scene/ModelImporter.h"

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>
#include <glm/vec3.hpp>

#include <algorithm>
#include <iostream>
#include <memory>

class CustomAssimpLogStream : public Assimp::LogStream
{
  public:
    void write(const char* /*message*/) override
    {
        // std::cout << message << std::endl;
    }
};

class AssimpLogging
{
  public:
    AssimpLogging()
    {
        Assimp::DefaultLogger::create();
        Assimp::DefaultLogger::get()->attachStream(&custom_log_stream_, severity_);
    }

    ~AssimpLogging()
    {
        Assimp::DefaultLogger::get()->detatchStream(&custom_log_stream_, severity_);
        Assimp::DefaultLogger::kill();
    }

    AssimpLogging(const AssimpLogging&) = delete;
    AssimpLogging(AssimpLogging&&) = delete;
    AssimpLogging& operator=(const AssimpLogging&) = delete;
    AssimpLogging& operator=(AssimpLogging&&) = delete;

  private:
    CustomAssimpLogStream custom_log_stream_{};
    const std::uint32_t severity_ =
        Assimp::Logger::Debugging | Assimp::Logger::Info | Assimp::Logger::Err | Assimp::Logger::Warn; // NOLINT
};

std::unique_ptr<aiScene> importAssimpScene(const std::string& file_name)
{
    AssimpLogging assimp_logging{};

    Assimp::Importer importer;
    importer.SetExtraVerbose(true);
    importer.ReadFile(file_name, aiProcess_Triangulate
                      //| aiProcess_JoinIdenticalVertices
                      //| aiProcess_SortByPType
                      //| aiProcess_PreTransformVertices
                      //| aiProcess_GenSmoothNormals		// Too slow
                      //| aiProcess_GenNormals
                      //| aiProcess_OptimizeMeshes
                      //| aiProcess_ImproveCacheLocality
    );

    std::unique_ptr<aiScene> assimp_scene{importer.GetOrphanedScene()};
    if (assimp_scene == nullptr)
    {
        std::cout << importer.GetErrorString() << std::endl;
        return {nullptr};
    }

    return assimp_scene;
}

aiMatrix4x4 getLocalToWorldTransform(const aiNode* node)
{
    std::vector<aiMatrix4x4> transforms;

    while (node != nullptr)
    {
        transforms.push_back(node->mTransformation);
        node = node->mParent;
    }

    aiMatrix4x4 local_to_world{};

    std::for_each(transforms.rbegin(), transforms.rend(), [&](const auto& transform) {
        local_to_world *= transform;
    });

    return local_to_world;
}

void getTotalNumvberOfTrianglesAndVertices(const aiScene* scene,
                                           const aiNode* node,
                                           std::uint64_t* n_triangles,
                                           std::uint64_t* n_vertices)
{
    for (std::uint32_t i = 0; i < node->mNumChildren; i++)
    {
        getTotalNumvberOfTrianglesAndVertices(scene, node->mChildren[i], n_triangles, n_vertices);
    }

    for (std::uint32_t i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        *n_triangles += mesh->mNumFaces;
        *n_vertices += mesh->mNumVertices;
    }
}

void printSceneInfo(const aiScene* scene)
{
    std::cout << "Scene info" << std::endl;
    std::cout << "\tMeshes:    " << scene->mNumMeshes << std::endl;

    std::uint64_t triangles = 0;
    std::uint64_t vertices = 0;
    // Count triangles and vertices
    for (std::uint32_t i = 0; i < scene->mNumMeshes; i++)
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
    std::uint64_t triangles = 0;
    std::uint64_t vertices = 0;
    // Count triangles and vertices
    for (std::uint32_t i = 0; i < scene->mNumMeshes; i++)
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
void computeAbsoluteTransform(aiNode* pc_node)
{
    if (pc_node->mParent != nullptr)
    {
        pc_node->mTransformation = pc_node->mParent->mTransformation * pc_node->mTransformation;
    }

    for (std::uint32_t i = 0; i < pc_node->mNumChildren; ++i)
    {
        computeAbsoluteTransform(pc_node->mChildren[i]);
    }
}

// If nodes have been transformed before hand
void convertToMeshRecursive(Scene& s, const aiScene* scene, const aiNode* node, Mesh* m, std::uint32_t& offset)
{
    aiMatrix4x4 m_world_it = node->mTransformation;
    m_world_it.Inverse().Transpose();
    aiMatrix3x3 m3x3{m_world_it};

    for (std::size_t i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // Copy vertex and normal data
        for (std::uint32_t k = 0; k < mesh->mNumVertices; k++)
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
        for (std::uint32_t j = 0; j < mesh->mNumFaces; j++)
        {
            // Copy triangle data
            aiFace* face = &mesh->mFaces[j];
            if (face->mNumIndices != 3) // if the face is not a triangle
            {
                continue;
            }
            Triangle t;
            for (std::uint32_t k = 0; k < face->mNumIndices; k++)
            {
                t.v[k] = face->mIndices[k] + offset;
                if (mesh->HasNormals())
                {
                    t.n[k] = face->mIndices[k] + offset;
                }
            }

            t.vdata = &m->v;
            t.ndata = &m->n;

            // Material
            t.material = &s.materials[mesh->mMaterialIndex];

            m->t.emplace_back(t);
        }

        offset += mesh->mNumVertices;

        if (!mesh->HasNormals())
        {
            m->genSmoothNormals();
        }
    }

    for (std::size_t i = 0; i < node->mNumChildren; i++)
    {
        convertToMeshRecursive(s, scene, node->mChildren[i], m, offset);
    }
}

void convertToMeshRecursive(Scene& s,
                            const aiScene* scene,
                            const aiNode* node,
                            aiMatrix4x4 transform,
                            Mesh* m,
                            uint32_t& offset)
{
    transform = transform * node->mTransformation;

    aiMatrix4x4 m_world_it = transform;
    m_world_it.Inverse().Transpose();
    aiMatrix3x3 m3x3{m_world_it};

    /*aiQuaternion quat;
    aiVector3D scale;
    aiVector3D pos;
    transform.Decompose(scale, quat, pos);*/

    for (std::size_t i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];

        // Copy vertex and normal data
        for (std::uint32_t k = 0; k < mesh->mNumVertices; k++)
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
        for (std::uint32_t j = 0; j < mesh->mNumFaces; j++)
        {
            // Copy triangle data
            aiFace* face = &mesh->mFaces[j];
            if (face->mNumIndices != 3) // if the face is not a triangle
            {
                continue;
            }
            Triangle t;
            for (std::uint32_t k = 0; k < face->mNumIndices; k++)
            {
                t.v[k] = face->mIndices[k] + offset;
                if (mesh->HasNormals())
                {
                    t.n[k] = face->mIndices[k] + offset;
                }
            }

            t.vdata = &m->v;
            t.ndata = &m->n;

            // Material
            t.material = &s.materials[mesh->mMaterialIndex];

            m->t.emplace_back(t);
        }

        offset += mesh->mNumVertices;

        if (!mesh->HasNormals())
        {
            m->genSmoothNormals();
        }
    }

    for (size_t i = 0; i < node->mNumChildren; i++)
    {
        convertToMeshRecursive(s, scene, node->mChildren[i], transform, m, offset);
    }
}

std::unique_ptr<Mesh> convertToMesh(const aiScene* scene, Scene& s)
{
    std::uint64_t triangles = 0;
    std::uint64_t vertices = 0;

    getTotalNumvberOfTrianglesAndVertices(scene, scene->mRootNode, &triangles, &vertices);

    auto m = std::make_unique<Mesh>();
    // Reserve memory
    m->t.reserve(triangles);
    m->v.reserve(vertices);
    m->n.reserve(vertices);

    std::uint32_t offset = 0;

    // ComputeAbsoluteTransform(scene->mRootNode);

    convertToMeshRecursive(s, scene, scene->mRootNode, aiMatrix4x4(), m.get(), offset);

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

    for (std::uint32_t j = 0; j < mat->mNumProperties; j++)
    {
        aiMaterialProperty* prop = mat->mProperties[j];

        std::cout << prop->mKey.C_Str() << ": ";

        aiPropertyTypeInfo type = prop->mType;

        switch (type)
        {
        case aiPTI_Float: {
            auto* farr = reinterpret_cast<float*>(prop->mData); // NOLINT
            for (std::size_t i = 0; i < prop->mDataLength / sizeof(float); i++)
            {
                std::cout << farr[i] << ", ";
            }
            std::cout << std::endl;
            break;
        }
        case aiPTI_Double: {
            auto* darr = reinterpret_cast<double*>(prop->mData); // NOLINT
            for (std::size_t i = 0; i < prop->mDataLength / sizeof(double); i++)
            {
                std::cout << darr[i] << ", ";
            }
            std::cout << std::endl;
            break;
        }
        case aiPTI_String: {
            // Extracted from assimp source:
            // ai_assert(prop->mDataLength >= 5);

            //// The string is stored as 32 but length prefix followed by zero-terminated UTF8 data
            // pOut->length = static_cast<std::uint32_t>(*reinterpret_cast<uint32_t*>(prop->mData));

            // ai_assert(pOut->length + 1 + 4 == prop->mDataLength);
            // ai_assert(!prop->mData[prop->mDataLength - 1]);
            // memcpy(pOut->data, prop->mData + 4, pOut->length + 1);

            for (size_t i = 4; i < prop->mDataLength; i++)
            {
                std::cout << prop->mData[i];
            }
            std::cout << std::endl;
            break;
        }
        case aiPTI_Integer: {
            auto* iarr = reinterpret_cast<int*>(prop->mData); // NOLINT
            for (std::size_t i = 0; i < prop->mDataLength / sizeof(int); i++)
            {
                std::cout << iarr[i] << ", ";
            }
            std::cout << std::endl;
            break;
        }
        case aiPTI_Buffer:
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

    for (std::uint32_t i = 0; i < aiscene->mNumMaterials; i++)
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
        {
            m.name = std::string(str.C_Str());
        }

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

        aiMatrix4x4 node_transform = getLocalToWorldTransform(nd);

        // multiply all properties of the camera with the absolute
        // transformation of the corresponding node
        /*aiVector3D camPos = nodeTransform * cam->mPosition;
        aiVector3D camLookAt = aiMatrix3x3(nodeTransform) * cam->mLookAt;
        aiVector3D camUp = aiMatrix3x3(nodeTransform) * cam->mUp;*/

        aiMatrix4x4 m_world_it = node_transform;
        m_world_it.Inverse().Transpose();
        aiMatrix3x3 m3x3{m_world_it};

        // multiply all properties of the camera with the absolute
        // transformation of the corresponding node
        aiVector3D cam_pos = node_transform * cam->mPosition;
        aiVector3D cam_look_at = m3x3 * cam->mLookAt;
        aiVector3D cam_up = m3x3 * cam->mUp;

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

        s.camera.eye = glm::vec3(cam_pos.x, cam_pos.y, cam_pos.z);
        s.camera.forward = glm::normalize(glm::vec3(cam_look_at.x, cam_look_at.y, cam_look_at.z));
        s.camera.up = glm::normalize(glm::vec3(cam_up.x, cam_up.y, cam_up.z));
        s.camera.right = glm::normalize(glm::cross(s.camera.forward, s.camera.up));

        // tan(FOV/2) = (screenSize/2) / screenPlaneDistance
        // tan(FOV_H/2) = (screen_width/2) / screenPlaneDistance
        // tan(FOV_V / 2) = (screen_height / 2) / screenPlaneDistance
        // tan(FOV_H/2) / screen_width = tan(FOV_V/2) / screen_height
        s.camera.aspect_ratio = cam->mAspect;
        s.camera.width = 640;
        s.camera.d = ((float)s.camera.width / 2.0f) / tanf(cam->mHorizontalFOV / 2.0f);
        s.camera.height = static_cast<int>((float)s.camera.width / cam->mAspect);
        s.camera.horizontalFOV(cam->mHorizontalFOV);

        // s.camera.computeUVW();
    }
    else
    {
        s.camera.fit(s.getBoundingBox());
    }

    return true;
}

void ModelImporter::importcbscene(const std::string& file_name, Scene& s, const std::function<void()>& cb)
{
    import(file_name, s);
    cb();
}

bool ModelImporter::import(const std::string& file_name, Scene& s)
{
    std::cout << "----------------------------<Importer>---------------------------" << std::endl;
    std::cout << "Assimp " << aiGetVersionMajor() << "." << aiGetVersionMinor() << std::endl;
    std::cout << "Loading scene..." << std::endl;

    const std::unique_ptr<aiScene> assimp_scene = importAssimpScene(file_name);
    if (assimp_scene == nullptr)
    {
        std::cout << "Assimp failed to load the file!" << std::endl;
        std::cout << "---------------------------</Importer>---------------------------" << std::endl;
        return false;
    }
    std::cout << "Done!" << std::endl << std::endl;

    printSceneInfo(assimp_scene.get());

    std::cout << "Converting scene..." << std::endl;
    convert(assimp_scene.get(), s);
    std::cout << "Done!" << std::endl << std::endl;

    std::cout << "Import of scene " << file_name.c_str() << " succeeded." << std::endl;
    std::cout << "---------------------------</Importer>---------------------------" << std::endl;
    return true;
}