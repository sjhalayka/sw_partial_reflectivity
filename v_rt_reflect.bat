glslc.exe "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\closesthit.rchit"  --target-env=vulkan1.3 -o "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\closesthit.rchit.spv" 

glslc.exe "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\miss.rmiss" --target-env=vulkan1.3 -o "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\miss.rmiss.spv"

glslc.exe "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\raygen.rgen" --target-env=vulkan1.3 -o "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\raygen.rgen.spv"

glslc.exe "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\shadow.rmiss" --target-env=vulkan1.3 -o "C:\dev\sw_vulkan_master\data\shaders\glsl\raytracingreflections\shadow.rmiss.spv"





raytracingreflections.exe


