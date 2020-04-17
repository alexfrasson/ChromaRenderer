#include "Object.h"
//#include "ObjLoader.h"

Object::Object()
{
}

size_t Object::sizeInBytes()
{
    size_t size = sizeof(Object);
    size += sizeof(Face) * f.size();
    return size;
}

void Object::genBoundingBox()
{
    size_t size = f.size();

    // find boundaries
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            if (f[i].v[j].x > boundingBox.max.x)
                boundingBox.max.x = f[i].v[j].x;
            if (f[i].v[j].y > boundingBox.max.y)
                boundingBox.max.y = f[i].v[j].y;
            if (f[i].v[j].z > boundingBox.max.z)
                boundingBox.max.z = f[i].v[j].z;
            if (f[i].v[j].x < boundingBox.min.x)
                boundingBox.min.x = f[i].v[j].x;
            if (f[i].v[j].y < boundingBox.min.y)
                boundingBox.min.y = f[i].v[j].y;
            if (f[i].v[j].z < boundingBox.min.z)
                boundingBox.min.z = f[i].v[j].z;
        }
    }
}

// void Object::load(std::string file)
// {

//     _ObjMesh m;
//     m.load(file);

//     //Copy vertices
//     //for(int i = 0; i < m.v.size(); i++)
//         //v.push_back(glm::vec3(m.v[i].x, m.v[i].y, m.v[i].z));
//         //v.emplace_back(m.v[i].x, m.v[i].y, m.v[i].z);
//     // v.resize(m.v.size());
//     // for(int i = 0; i < m.v.size(); i++)
//     // {
//     //     v[i].x = m.v[i].x;
//     //     v[i].y = m.v[i].y;
//     //     v[i].z = m.v[i].z;
//     // }
// Copy normals
// for(int i = 0; i < m.n.size(); i++)
//	n.push_back(glm::vec3(m.n[i].x, m.n[i].y, m.n[i].z));

// Copy indexes
// f.resize(m.f.size());
// for(int i = 0; i < (int)m.f.size(); i++){
//     for(int j = 0; j < 3; j++){
//         //v
//         f[i].v[j] = &v[m.f[i].v[j]];
//         //n
//         f[i].n[j].x = m.n[m.f[i].n[j]].x;
//         f[i].n[j].y = m.n[m.f[i].n[j]].y;
//         f[i].n[j].z = m.n[m.f[i].n[j]].z;
//     }

//     //calcula normal do plano
//     f[i].tn = glm::cross((*f[i].v[1] - *f[i].v[0]), (*f[i].v[2] - *f[i].v[0]));
//     f[i].tn = glm::normalize(f[i].tn);

//     //mollertrumbore edges
//     f[i].edgesMollerTrumbore[0] = *f[i].v[1] - *f[i].v[0];
//     f[i].edgesMollerTrumbore[1] = *f[i].v[2] - *f[i].v[0];
// }
// f.resize(m.f.size());
// for(int i = 0; i < (int)m.f.size(); i++){
//     for(int j = 0; j < 3; j++){
//         //v
//         f[i].v[j].x = m.v[m.f[i].v[j]].x;
//         f[i].v[j].y = m.v[m.f[i].v[j]].y;
//         f[i].v[j].z = m.v[m.f[i].v[j]].z;
//         //n
//         f[i].n[j].x = m.n[m.f[i].n[j]].x;
//         f[i].n[j].y = m.n[m.f[i].n[j]].y;
//         f[i].n[j].z = m.n[m.f[i].n[j]].z;
//     }

//     //calcula normal do plano
//     f[i].tn = glm::cross((f[i].v[1] - f[i].v[0]), (f[i].v[2] - f[i].v[0]));
//     f[i].tn = glm::normalize(f[i].tn);

//     //mollertrumbore edges
//     f[i].edgesMollerTrumbore[0] = f[i].v[1] - f[i].v[0];
//     f[i].edgesMollerTrumbore[1] = f[i].v[2] - f[i].v[0];
// }


// boundingBox.genBoundingBox(m);
// }