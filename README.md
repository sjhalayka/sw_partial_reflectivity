Based off of Sascha Willems' raytracingreflections demo code.

![image](https://github.com/sjhalayka/sw_partial_reflectivity/assets/16404554/9e81e3f3-5447-4c69-927b-011a24adb79e)

Supports:
- Reflection and refraction (Fresnel)
- Chromatic aberration
- Multiple shadow-casting lights
- Multiple textures
- Fast glossy (noisy) reflections
- Fast blurry (noisy) shadow edges
- Volumetric fog via ray marching/tracing
- Caustic refraction
- Large-format screenshots (may require a TDR time extension, if your framerate dips below 1 FPS, like on my 3060):

KeyPath   : HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers<br>
KeyValue  : TdrDelay<br>
ValueType : REG_DWORD (32bit)<br>
ValueData : Number of seconds to delay. 2 seconds is the default value.<br>

Soon to support:
- Depth of field effects
- 3D texture lookup for custom volumetric effects (like, smoke from EmberGen).
- Caustic reflections

The fractal_500.gltf file can be downloaded from: https://drive.google.com/file/d/1BJJSC_K8NwaH8kP4tQpxlAmc6h6N3Ii1/view
