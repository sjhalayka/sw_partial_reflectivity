Based off of Sascha Willems' raytracingreflections demo code.

![image](https://github.com/sjhalayka/sw_partial_reflectivity/assets/16404554/5e52bddc-5fab-4dee-bfad-5e8823bfefe2)

Supports:
- Reflection and refraction (Fresnel)
- Chromatic aberration
- Multiple shadow-casting lights
- Multiple textures
- Fast glossy (noisy) reflections
- Fast blurry (noisy) shadow edges
- Volumetric fog via ray marching/tracing
- Large-format screenshots (requires a TDR extension):

KeyPath   : HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
KeyValue  : TdrDelay
ValueType : REG_DWORD (32bit)
ValueData : Number of seconds to delay. 2 seconds is the default value.

Soon to support:
- Depth of field effects
- 3D texture lookup for custom volumetric effects (like, smoke from EmberGen).

The fractal_500.gltf file can be downloaded from: https://drive.google.com/file/d/1BJJSC_K8NwaH8kP4tQpxlAmc6h6N3Ii1/view
