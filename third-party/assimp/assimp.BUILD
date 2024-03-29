load("@rules_cc//cc:defs.bzl", "cc_library")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@//third-party/expand_template:expand_template.bzl", "expand_template")

EXCLUDED_IMPORTERS = [
    #"ASSIMP_BUILD_NO_COLLADA_IMPORTER",
    #"ASSIMP_BUILD_NO_OBJ_IMPORTER",
    #"ASSIMP_BUILD_NO_GLTF_IMPORTER",
    "ASSIMP_BUILD_NO_C4D_IMPORTER",
    "ASSIMP_BUILD_NO_AMF_IMPORTER",
    "ASSIMP_BUILD_NO_3DS_IMPORTER",
    "ASSIMP_BUILD_NO_AC_IMPORTER",
    "ASSIMP_BUILD_NO_ASE_IMPORTER",
    "ASSIMP_BUILD_NO_ASSBIN_IMPORTER",
    "ASSIMP_BUILD_NO_B3D_IMPORTER",
    "ASSIMP_BUILD_NO_BVH_IMPORTER",
    "ASSIMP_BUILD_NO_DXF_IMPORTER",
    "ASSIMP_BUILD_NO_CSM_IMPORTER",
    "ASSIMP_BUILD_NO_HMP_IMPORTER",
    "ASSIMP_BUILD_NO_IRRMESH_IMPORTER",
    "ASSIMP_BUILD_NO_IRR_IMPORTER",
    "ASSIMP_BUILD_NO_LWO_IMPORTER",
    "ASSIMP_BUILD_NO_LWS_IMPORTER",
    "ASSIMP_BUILD_NO_MD2_IMPORTER",
    "ASSIMP_BUILD_NO_MD3_IMPORTER",
    "ASSIMP_BUILD_NO_MD5_IMPORTER",
    "ASSIMP_BUILD_NO_MDC_IMPORTER",
    "ASSIMP_BUILD_NO_MDL_IMPORTER",
    "ASSIMP_BUILD_NO_NFF_IMPORTER",
    "ASSIMP_BUILD_NO_NDO_IMPORTER",
    "ASSIMP_BUILD_NO_OFF_IMPORTER",
    "ASSIMP_BUILD_NO_OGRE_IMPORTER",
    "ASSIMP_BUILD_NO_OPENGEX_IMPORTER",
    "ASSIMP_BUILD_NO_PLY_IMPORTER",
    "ASSIMP_BUILD_NO_MS3D_IMPORTER",
    "ASSIMP_BUILD_NO_COB_IMPORTER",
    "ASSIMP_BUILD_NO_BLEND_IMPORTER",
    "ASSIMP_BUILD_NO_IFC_IMPORTER",
    "ASSIMP_BUILD_NO_XGL_IMPORTER",
    "ASSIMP_BUILD_NO_FBX_IMPORTER",
    "ASSIMP_BUILD_NO_Q3D_IMPORTER",
    "ASSIMP_BUILD_NO_Q3BSP_IMPORTER",
    "ASSIMP_BUILD_NO_RAW_IMPORTER",
    "ASSIMP_BUILD_NO_SIB_IMPORTER",
    "ASSIMP_BUILD_NO_SMD_IMPORTER",
    "ASSIMP_BUILD_NO_STL_IMPORTER",
    "ASSIMP_BUILD_NO_TERRAGEN_IMPORTER",
    "ASSIMP_BUILD_NO_3D_IMPORTER",
    "ASSIMP_BUILD_NO_X_IMPORTER",
    "ASSIMP_BUILD_NO_X3D_IMPORTER",
    "ASSIMP_BUILD_NO_3MF_IMPORTER",
    "ASSIMP_BUILD_NO_MMD_IMPORTER",
    "ASSIMP_BUILD_NO_STEP_IMPORTER",
]

EXCLUDED_EXPORTERS = [
    "ASSIMP_BUILD_NO_EXPORT",
    # "ASSIMP_BUILD_NO_COLLADA_EXPORTER",
    # "ASSIMP_BUILD_NO_3DS_EXPORTER",
    # "ASSIMP_BUILD_NO_ASSBIN_EXPORTER",
    # "ASSIMP_BUILD_NO_ASSXML_EXPORTER",
    # "ASSIMP_BUILD_NO_OBJ_EXPORTER",
    # "ASSIMP_BUILD_NO_OPENGEX_EXPORTER",
    # "ASSIMP_BUILD_NO_PLY_EXPORTER",
    # "ASSIMP_BUILD_NO_FBX_EXPORTER",
    # "ASSIMP_BUILD_NO_STL_EXPORTER",
    # "ASSIMP_BUILD_NO_X_EXPORTER",
    # "ASSIMP_BUILD_NO_X3D_EXPORTER",
    # "ASSIMP_BUILD_NO_GLTF_EXPORTER",
    # "ASSIMP_BUILD_NO_3MF_EXPORTER",
    # "ASSIMP_BUILD_NO_ASSJSON_EXPORTER",
    # "ASSIMP_BUILD_NO_STEP_EXPORTER",
]

cc_library(
    name = "assimp",
    visibility = ["//visibility:public"],
    deps = [
        ":capi",
        ":collada_importer",
        ":common",
        ":gltf_importer",
        ":material",
        ":obj_importer",
        ":post_processing",
    ],
)

cc_library(
    name = "common",
    srcs = glob(["code/Common/*.cpp"]),
    defines = EXCLUDED_IMPORTERS + EXCLUDED_EXPORTERS,
    linkstatic = True,
    deps = [
        ":common_headers",
        ":irrxml",
        ":public_headers",
        ":unzip",
        ":utf8cpp",
    ],
    alwayslink = True,
)

cc_library(
    name = "common_headers",
    hdrs = glob(["code/Common/*.h"]),
    includes = ["code"],
)

cc_library(
    name = "public_headers",
    hdrs = glob([
        "include/*.h",
        "include/*.hpp",
        "include/*.inl",
        "include/Compiler/*.h",
    ]) + [
        ":config_tmpl",
        ":revision_tmpl",
    ],
    includes = ["include"],
)

cc_library(
    name = "material",
    srcs = ["code/Material/MaterialSystem.cpp"],
    hdrs = ["code/Material/MaterialSystem.h"],
    deps = [":public_headers"],
)

cc_library(
    name = "post_processing",
    srcs = glob(["code/PostProcessing/*.cpp"]),
    hdrs = glob(["code/PostProcessing/*.h"]),
    deps = [
        ":common_headers",
        ":public_headers",
    ],
)

cc_library(
    name = "capi",
    srcs = glob(["code/CApi/*.cpp"]),
    hdrs = glob(["code/CApi/*.h"]),
    linkstatic = True,
    deps = [
        ":common_headers",
        ":public_headers",
    ],
)

########################################################################################################################
# templates
########################################################################################################################

expand_template(
    name = "revision_tmpl",
    out = "revision.h",
    substitutions = {
        "@GIT_COMMIT_HASH@": "0",
        "@GIT_BRANCH@": "master",
    },
    template = "revision.h.in",
)

expand_template(
    name = "config_tmpl",
    out = "include/assimp/config.h",
    substitutions = {"#cmakedefine": "// #undef"},
    template = "include/assimp/config.h.in",
)

########################################################################################################################
# importers
########################################################################################################################

cc_library(
    name = "collada_importer",
    srcs = ["code/Collada/ColladaLoader.cpp"],
    hdrs = ["code/Collada/ColladaLoader.h"],
    includes = ["code/Collada"],
    deps = [":collada_parser"],
    alwayslink = True,
)

cc_library(
    name = "obj_importer",
    srcs = [
        "code/Obj/ObjFileImporter.cpp",
        "code/Obj/ObjFileMtlImporter.cpp",
        "code/Obj/ObjFileParser.cpp",
    ],
    hdrs = [
        "code/Obj/ObjFileData.h",
        "code/Obj/ObjFileImporter.h",
        "code/Obj/ObjFileMtlImporter.h",
        "code/Obj/ObjFileParser.h",
        "code/Obj/ObjTools.h",
    ],
    deps = [":public_headers"],
    alwayslink = True,
)

cc_library(
    name = "gltf_importer",
    srcs = [
        "code/glTF/glTFCommon.cpp",
        "code/glTF/glTFImporter.cpp",
        "code/glTF2/glTF2Importer.cpp",
    ],
    hdrs = [
        "code/glTF/glTFCommon.h",
        "code/glTF/glTFImporter.h",
        "code/glTF2/glTF2Importer.h",
    ],
    deps = [
        ":common_headers",
        ":public_headers",
        ":rapidjson",
    ],
    alwayslink = True,
)

########################################################################################################################
# exporters
########################################################################################################################

cc_library(
    name = "collada_exporter",
    srcs = ["code/Collada/ColladaExporter.cpp"],
    hdrs = ["code/Collada/ColladaExporter.h"],
    includes = ["code/Collada"],
    deps = [":collada_parser"],
    alwayslink = True,
)

########################################################################################################################
# importer/exporter common
########################################################################################################################

cc_library(
    name = "collada_parser",
    srcs = ["code/Collada/ColladaParser.cpp"],
    hdrs = [
        "code/Collada/ColladaHelper.h",
        "code/Collada/ColladaParser.h",
    ],
    includes = ["code/Collada"],
    deps = [
        ":irrxml",
        ":public_headers",
    ],
    alwayslink = True,
)

########################################################################################################################
# contrib (i.e. third_party)
########################################################################################################################

cc_library(
    name = "unzip",
    srcs = glob(["contrib/unzip/*.c"]),
    hdrs = glob(["contrib/unzip/*.h"]),
    includes = ["contrib/unzip"],
    deps = [":zlib"],
)

copy_file(
    name = "zconf_header",
    src = "contrib/zlib/zconf.h.included",
    out = "zconf.h",
)

cc_library(
    name = "zlib",
    srcs = glob(["contrib/zlib/*.c"]),
    hdrs = glob(["contrib/zlib/*.h"]) + [":zconf_header"],
    includes = ["contrib/zlib"],
)

cc_library(
    name = "irrxml",
    srcs = ["contrib/irrXML/irrXML.cpp"],
    hdrs = glob([
        "contrib/irrXML/*.h",
        "contrib/irrXML/*.hpp",
    ]),
    includes = ["contrib/irrXML"],
)

cc_library(
    name = "utf8cpp",
    srcs = glob(["contrib/utf8cpp/*.c"]),
    hdrs = glob(["contrib/utf8cpp/*.h"]),
    includes = ["contrib/utf8cpp"],
)

cc_library(
    name = "rapidjson",
    hdrs = glob(["contrib/rapidjson/include/**/*.h"]),
    includes = ["contrib/rapidjson/include"],
)
