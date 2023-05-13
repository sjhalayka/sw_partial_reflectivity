#include "VulkanRaytracingSample.h"
#include "VulkanglTFModel.h"

#include <vector>
using std::vector;

#include <fstream>
using std::ofstream;

#include <ios>
using std::ios;

#include <mutex>
using std::mutex;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


class VulkanExample : public VulkanRaytracingSample
{
public:
	AccelerationStructure bottomLevelAS{};
	AccelerationStructure topLevelAS{};

	bool do_screenshot = false;
	//mutex m;

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_SPACE:
		{
			if (prepared)
				screenshot(width * 6, height * 6, "v_rt_reflect.png");

			break;
		}
		}
	}


	void screenshot(const uint32_t size_x, const uint32_t size_y, const char* filename)
	{
		//m.lock();

		const VkDeviceSize size = size_x * size_y * 4; // number of bytes

		// Create screenshot image
		createScreenshotStorageImage(VK_FORMAT_R8G8B8A8_UNORM, { size_x, size_y, 1 });

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&screenshotStagingBuffer,
			size
		));

		// Update descriptor
		{
			VkDescriptorImageInfo storageImageDescriptor{ VK_NULL_HANDLE, screenshotStorageImage.view, VK_IMAGE_LAYOUT_GENERAL };
			VkWriteDescriptorSet resultImageWrite = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor);
			vkUpdateDescriptorSets(device, 1, &resultImageWrite, 0, VK_NULL_HANDLE);
		}

		// Prepare & flush command buffer
		{
			VkCommandBuffer screenshotCmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

			VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			VkDescriptorSet sets[] = { descriptorSet, scene.materials[0].descriptorSet };

			vkCmdBindPipeline(screenshotCmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
			vkCmdBindDescriptorSets(screenshotCmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 2, sets, 0, 0);

			VkStridedDeviceAddressRegionKHR emptySbtEntry = {};
			vkCmdTraceRaysKHR(
				screenshotCmdBuffer,
				&shaderBindingTables.raygen.stridedDeviceAddressRegion,
				&shaderBindingTables.miss.stridedDeviceAddressRegion,
				&shaderBindingTables.hit.stridedDeviceAddressRegion,
				&emptySbtEntry,
				size_x,
				size_y,
				1);



			vks::tools::setImageLayout(
				screenshotCmdBuffer,
				screenshotStorageImage.image,
				VK_IMAGE_LAYOUT_GENERAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				subresourceRange);



			//vks::tools::insertImageMemoryBarrier(
			//	screenshotCmdBuffer,
			//	screenshotStorageImage.image,
			//	0,
			//	VK_ACCESS_TRANSFER_WRITE_BIT,
			//	VK_IMAGE_LAYOUT_UNDEFINED,
			//	VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });







			VkBufferImageCopy copyRegion{};
			copyRegion.bufferOffset = 0;
			copyRegion.bufferRowLength = 0;
			copyRegion.bufferImageHeight = 0;

			copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copyRegion.imageSubresource.mipLevel = 0;
			copyRegion.imageSubresource.baseArrayLayer = 0;
			copyRegion.imageSubresource.layerCount = 1;

			copyRegion.imageOffset = { 0, 0, 0 };
			copyRegion.imageExtent.width = size_x;
			copyRegion.imageExtent.height = size_y;
			copyRegion.imageExtent.depth = 1;

			vkCmdCopyImageToBuffer(screenshotCmdBuffer, screenshotStorageImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, screenshotStagingBuffer.buffer, 1, &copyRegion);

			vulkanDevice->flushCommandBuffer(screenshotCmdBuffer, queue);
		}

		// Copy pixel data
		VK_CHECK_RESULT(screenshotStagingBuffer.map());

		vector<uint8_t> data(size);
		memcpy(&data[0], screenshotStagingBuffer.mapped, size);

		for (uint32_t i = 0; i < size_x; i++)
		{
			for (uint32_t j = 0; j < size_y; j++)
			{
				size_t index = 4 * (j * size_x + i);

				//unsigned char temp_char;
				//temp_char = data[index + 0];
				//data[index + 0] = data[index + 2];
				//data[index + 2] = temp_char;
				data[index + 3] = 255;
			}
		}

		//	if (1)//true == reverse_rows)
		//	{
		//		// Reverse row order
		//		short unsigned int num_rows_to_swap = py;
		//		vector<unsigned char> buffer(px * 4);

		//		if (0 != py % 2)
		//			num_rows_to_swap--;

		//		num_rows_to_swap /= 2;

		//		for (short unsigned int i = 0; i < num_rows_to_swap; i++)
		//		{
		//			size_t y_first = i * px * 4;
		//			size_t y_last = (py - 1 - i) * px * 4;

		//			memcpy(&buffer[0], &data[y_first], px * 4 * sizeof(unsigned char));
		//			memcpy(&data[y_first], &data[y_last], px * 4 * sizeof(unsigned char));
		//			memcpy(&data[y_last], &buffer[0], px * 4 * sizeof(unsigned char));
		//		}
		//	}





			// Set up Targa TGA image data.
			//unsigned char  idlength = 0;
			//unsigned char  colourmaptype = 0;
			//unsigned char  datatypecode = 2;
			//unsigned short int colourmaporigin = 0;
			//unsigned short int colourmaplength = 0;
			//unsigned char  colourmapdepth = 0;
			//unsigned short int x_origin = 0;
			//unsigned short int y_origin = 0;

			//unsigned short int px = size_x;
			//unsigned short int py = size_y;
			//unsigned char  bitsperpixel = 32;
			//unsigned char  imagedescriptor = 0;
			//vector<char> idstring;



		//	// Write Targa TGA file to disk.
		//	ofstream out(filename, ios::binary);

		//	if (!out.is_open())
		//	{
		////		cout << "Failed to open TGA file for writing: " << filename << endl;
		//		return;
		//	}

		//	out.write(reinterpret_cast<char*>(&idlength), 1);
		//	out.write(reinterpret_cast<char*>(&colourmaptype), 1);
		//	out.write(reinterpret_cast<char*>(&datatypecode), 1);
		//	out.write(reinterpret_cast<char*>(&colourmaporigin), 2);
		//	out.write(reinterpret_cast<char*>(&colourmaplength), 2);
		//	out.write(reinterpret_cast<char*>(&colourmapdepth), 1);
		//	out.write(reinterpret_cast<char*>(&x_origin), 2);
		//	out.write(reinterpret_cast<char*>(&y_origin), 2);
		//	out.write(reinterpret_cast<char*>(&px), 2);
		//	out.write(reinterpret_cast<char*>(&py), 2);
		//	out.write(reinterpret_cast<char*>(&bitsperpixel), 1);
		//	out.write(reinterpret_cast<char*>(&imagedescriptor), 1);
		//	out.write(reinterpret_cast<char*>(&data[0]), size_x * size_y * 4 * sizeof(unsigned char));
		//	out.close();



		int result = stbi_write_png(filename, size_x, size_y, 4, &data[0], 0);




		// Update descriptor back to normal
		{
			VkDescriptorImageInfo storageImageDescriptor{ VK_NULL_HANDLE, storageImage.view, VK_IMAGE_LAYOUT_GENERAL };
			VkWriteDescriptorSet resultImageWrite = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor);
			vkUpdateDescriptorSets(device, 1, &resultImageWrite, 0, VK_NULL_HANDLE);
		}

		// Delete screenshot image
		deleteScreenshotStorageImage();
		screenshotStagingBuffer.destroy();


		//m.unlock();
	}


	void createScreenshotStorageImage(VkFormat format, VkExtent3D extent)
	{
		// Release ressources if image is to be recreated
		if (screenshotStorageImage.image != VK_NULL_HANDLE)
		{
			vkDestroyImageView(device, screenshotStorageImage.view, nullptr);
			vkDestroyImage(device, screenshotStorageImage.image, nullptr);
			vkFreeMemory(device, screenshotStorageImage.memory, nullptr);
			screenshotStorageImage = {};
		}

		VkImageCreateInfo image = vks::initializers::imageCreateInfo();
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = format;
		image.extent = extent;
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(vulkanDevice->logicalDevice, &image, nullptr, &screenshotStorageImage.image));

		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(vulkanDevice->logicalDevice, screenshotStorageImage.image, &memReqs);
		VkMemoryAllocateInfo memoryAllocateInfo = vks::initializers::memoryAllocateInfo();
		memoryAllocateInfo.allocationSize = memReqs.size;
		memoryAllocateInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(vulkanDevice->logicalDevice, &memoryAllocateInfo, nullptr, &screenshotStorageImage.memory));
		VK_CHECK_RESULT(vkBindImageMemory(vulkanDevice->logicalDevice, screenshotStorageImage.image, screenshotStorageImage.memory, 0));

		VkImageViewCreateInfo colorImageView = vks::initializers::imageViewCreateInfo();
		colorImageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		colorImageView.format = format;
		colorImageView.subresourceRange = {};
		colorImageView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		colorImageView.subresourceRange.baseMipLevel = 0;
		colorImageView.subresourceRange.levelCount = 1;
		colorImageView.subresourceRange.baseArrayLayer = 0;
		colorImageView.subresourceRange.layerCount = 1;
		colorImageView.image = screenshotStorageImage.image;
		VK_CHECK_RESULT(vkCreateImageView(vulkanDevice->logicalDevice, &colorImageView, nullptr, &screenshotStorageImage.view));

		VkCommandBuffer cmdBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vks::tools::setImageLayout(cmdBuffer, screenshotStorageImage.image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_GENERAL,
			{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		// Transition swapchain image from present to transfer source layout
		//vks::tools::insertImageMemoryBarrier(
		//	cmdBuffer,
		//	screenshotStorageImage.image,
		//	VK_ACCESS_MEMORY_READ_BIT,
		//	VK_ACCESS_TRANSFER_READ_BIT,
		//	VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		//	VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		//	VK_PIPELINE_STAGE_TRANSFER_BIT,
		//	VK_PIPELINE_STAGE_TRANSFER_BIT,
		//	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

		vulkanDevice->flushCommandBuffer(cmdBuffer, queue);
	}

	void deleteScreenshotStorageImage()
	{
		vkDestroyImageView(vulkanDevice->logicalDevice, screenshotStorageImage.view, nullptr);
		vkDestroyImage(vulkanDevice->logicalDevice, screenshotStorageImage.image, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, screenshotStorageImage.memory, nullptr);

		screenshotStorageImage = {};
	}






	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};
	struct ShaderBindingTables {
		ShaderBindingTable raygen;
		ShaderBindingTable miss;
		ShaderBindingTable hit;
	} shaderBindingTables;


	static const int max_lights = 2;

	VkTransformMatrixKHR transformMatrix = {
				1.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 0.0f };

	clock_t start = std::clock();

	struct UniformData {

		glm::mat4 viewInverse;
		glm::mat4 projInverse;
		glm::mat4 transformation_matrix;
		glm::vec4 light_positions[max_lights];
		glm::vec4 light_colors[max_lights];

		glm::vec3 camera_pos;

		int32_t vertexSize;

	} uniformData;
	vks::Buffer ubo;

	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSet descriptorSet;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSetLayout materialSetLayout;

	vkglTF::Model scene;

	VulkanRaytracingSample::StorageImage screenshotStorageImage;
	vks::Buffer screenshotStagingBuffer;

	// This sample is derived from an extended base class that saves most of the ray tracing setup boiler plate
	VulkanExample() : VulkanRaytracingSample()
	{
		title = "Ray tracing shadows & reflections";
		//timerSpeed *= 0.5f;
		//camera.rotationSpeed *= 0.25f;
		camera.type = Camera::CameraType::lookat;
		camera.setPerspective(45.0f, (float)width / (float)height, 0.001f, 10000.0f);
		camera.setRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		camera.setTranslation(glm::vec3(0.0f, 0.0f, -6.0));
		enableExtensions();
	}

	~VulkanExample()
	{
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, materialSetLayout, nullptr);
		deleteStorageImage();
		deleteAccelerationStructure(bottomLevelAS);
		deleteAccelerationStructure(topLevelAS);
		shaderBindingTables.raygen.destroy();
		shaderBindingTables.miss.destroy();
		shaderBindingTables.hit.destroy();
		ubo.destroy();
	}

	/*
		Create the bottom level acceleration structure contains the scene's actual geometry (vertices, triangles)
	*/
	void createBottomLevelAccelerationStructure()
	{
		// Instead of a simple triangle, we'll be loading a more complex scene for this example
		// The shaders are accessing the vertex and index buffers of the scene, so the proper usage flag has to be set on the vertex and index buffers for the scene
		vkglTF::memoryPropertyFlags = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		vkglTF::descriptorBindingFlags = vkglTF::DescriptorBindingFlags::ImageBaseColor | vkglTF::DescriptorBindingFlags::ImageNormalMap;
		const uint32_t glTFLoadingFlags = vkglTF::FileLoadingFlags::PreTransformVertices | vkglTF::FileLoadingFlags::PreMultiplyVertexColors | vkglTF::FileLoadingFlags::FlipY;

		//scene.loadFromFile(getAssetPath() + "models/reflection_scene.gltf", vulkanDevice, queue, glTFLoadingFlags);

		// This fractal_500.gltf file can be downloaded from:
		// https://drive.google.com/file/d/1BJJSC_K8NwaH8kP4tQpxlAmc6h6N3Ii1/view
		scene.loadFromFile("C:/temp/hires/fractal_500.gltf", vulkanDevice, queue, glTFLoadingFlags);

		//scene.loadFromFile("C:/temp/cylinder.gltf", vulkanDevice, queue, glTFLoadingFlags);


		//scene.loadFromFile(getAssetPath() + "models/FlightHelmet/glTF/FlightHelmet.gltf", vulkanDevice, queue, glTFLoadingFlags);

		VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
		VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

		vertexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.vertices.buffer);
		indexBufferDeviceAddress.deviceAddress = getBufferDeviceAddress(scene.indices.buffer);

		uint32_t numTriangles = static_cast<uint32_t>(scene.indices.count) / 3;
		uint32_t maxVertex = scene.vertices.count;

		// Build
		VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
		accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
		accelerationStructureGeometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
		accelerationStructureGeometry.geometry.triangles.maxVertex = maxVertex;
		accelerationStructureGeometry.geometry.triangles.vertexStride = sizeof(vkglTF::Vertex);
		accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
		accelerationStructureGeometry.geometry.triangles.indexData = indexBufferDeviceAddress;
		accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
		accelerationStructureGeometry.geometry.triangles.transformData.hostAddress = nullptr;

		// Get size info
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationStructureBuildGeometryInfo.geometryCount = 1;
		accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			&numTriangles,
			&accelerationStructureBuildSizesInfo);

		createAccelerationStructure(bottomLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, accelerationStructureBuildSizesInfo);

		// Create a small scratch buffer used during build of the bottom level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationBuildGeometryInfo.dstAccelerationStructure = bottomLevelAS.handle;
		accelerationBuildGeometryInfo.geometryCount = 1;
		accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
		accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

		VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
		accelerationStructureBuildRangeInfo.primitiveCount = numTriangles;
		accelerationStructureBuildRangeInfo.primitiveOffset = 0;
		accelerationStructureBuildRangeInfo.firstVertex = 0;
		accelerationStructureBuildRangeInfo.transformOffset = 0;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationBuildGeometryInfo,
			accelerationBuildStructureRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);

		deleteScratchBuffer(scratchBuffer);
	}

	/*
		The top level acceleration structure contains the scene's object instances
	*/
	void createTopLevelAccelerationStructure()
	{
		//static const float pi = 4.0f * atanf(1.0f);
		//float duration = (std::clock() - start) / (float) CLOCKS_PER_SEC;
		//float radians = duration * 2.0f * pi * 0.05f;

		//// Rotate on y axis
		//transformMatrix = {
		//	cos(radians), 0.0f, -sin(radians), 0.0f,
		//	0.0f,         1.0f,  0.0f,         0.0f,
		//	sin(radians), 0.0f,  cos(radians), 0.0f };

		VkAccelerationStructureInstanceKHR instance{};
		instance.transform = transformMatrix;
		instance.instanceCustomIndex = 0;
		instance.mask = 0xFF;
		instance.instanceShaderBindingTableRecordOffset = 0;
		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		instance.accelerationStructureReference = bottomLevelAS.deviceAddress;

		// Buffer for instance data
		vks::Buffer instancesBuffer;
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&instancesBuffer,
			sizeof(VkAccelerationStructureInstanceKHR),
			&instance));

		VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
		instanceDataDeviceAddress.deviceAddress = getBufferDeviceAddress(instancesBuffer.buffer);

		VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
		accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
		accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

		// Get size info
		VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR; //VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationStructureBuildGeometryInfo.geometryCount = 1;
		accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

		uint32_t primitive_count = 1;

		VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
		vkGetAccelerationStructureBuildSizesKHR(
			device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
			&accelerationStructureBuildGeometryInfo,
			&primitive_count,
			&accelerationStructureBuildSizesInfo);

		createAccelerationStructure(topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, accelerationStructureBuildSizesInfo);

		// Create a small scratch buffer used during build of the top level acceleration structure
		ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

		VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
		accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR; // VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		accelerationBuildGeometryInfo.dstAccelerationStructure = topLevelAS.handle;
		accelerationBuildGeometryInfo.geometryCount = 1;
		accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
		accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

		VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
		accelerationStructureBuildRangeInfo.primitiveCount = 1;
		accelerationStructureBuildRangeInfo.primitiveOffset = 0;
		accelerationStructureBuildRangeInfo.firstVertex = 0;
		accelerationStructureBuildRangeInfo.transformOffset = 0;
		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

		// Build the acceleration structure on the device via a one-time command buffer submission
		// Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
		VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		vkCmdBuildAccelerationStructuresKHR(
			commandBuffer,
			1,
			&accelerationBuildGeometryInfo,
			accelerationBuildStructureRangeInfos.data());
		vulkanDevice->flushCommandBuffer(commandBuffer, queue);

		deleteScratchBuffer(scratchBuffer);
		instancesBuffer.destroy();
	}

	/*
		Create the Shader Binding Tables that binds the programs and top-level acceleration structure

		SBT Layout used in this sample:

			/-----------\
			| raygen    |
			|-----------|
			| miss      |
			|-----------|
			| hit       |
			\-----------/

	*/
	void createShaderBindingTables() {
		const uint32_t handleSize = rayTracingPipelineProperties.shaderGroupHandleSize;
		const uint32_t handleSizeAligned = vks::tools::alignedSize(rayTracingPipelineProperties.shaderGroupHandleSize, rayTracingPipelineProperties.shaderGroupHandleAlignment);
		const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
		const uint32_t sbtSize = groupCount * handleSizeAligned;

		std::vector<uint8_t> shaderHandleStorage(sbtSize);
		VK_CHECK_RESULT(vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, groupCount, sbtSize, shaderHandleStorage.data()));

		createShaderBindingTable(shaderBindingTables.raygen, 1);
		createShaderBindingTable(shaderBindingTables.miss, 2);
		createShaderBindingTable(shaderBindingTables.hit, 1);

		// Copy handles
		memcpy(shaderBindingTables.raygen.mapped, shaderHandleStorage.data(), handleSize);
		memcpy(shaderBindingTables.miss.mapped, shaderHandleStorage.data() + handleSizeAligned, handleSize * 2);
		memcpy(shaderBindingTables.hit.mapped, shaderHandleStorage.data() + handleSizeAligned * 3, handleSize);
	}

	/*
		Create the descriptor sets used for the ray tracing dispatch
	*/
	void createDescriptorSets()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }
		};
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 1);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));

		VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

		VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = vks::initializers::writeDescriptorSetAccelerationStructureKHR();
		descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
		descriptorAccelerationStructureInfo.pAccelerationStructures = &topLevelAS.handle;

		VkWriteDescriptorSet accelerationStructureWrite{};
		accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		// The specialized acceleration structure descriptor has to be chained
		accelerationStructureWrite.pNext = &descriptorAccelerationStructureInfo;
		accelerationStructureWrite.dstSet = descriptorSet;
		accelerationStructureWrite.dstBinding = 0;
		accelerationStructureWrite.descriptorCount = 1;
		accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

		VkDescriptorImageInfo storageImageDescriptor{ VK_NULL_HANDLE, storageImage.view, VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorBufferInfo vertexBufferDescriptor{ scene.vertices.buffer, 0, VK_WHOLE_SIZE };
		VkDescriptorBufferInfo indexBufferDescriptor{ scene.indices.buffer, 0, VK_WHOLE_SIZE };

		std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
			// Binding 0: Top level acceleration structure
			accelerationStructureWrite,
			// Binding 1: Ray tracing result image
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor),
			// Binding 2: Uniform data
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &ubo.descriptor),
			// Binding 3: Scene vertex buffer
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &vertexBufferDescriptor),
			// Binding 4: Scene index buffer
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &indexBufferDescriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);
	}

	/*
		Create our ray tracing pipeline
	*/
	void createRayTracingPipeline()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0: Acceleration structure
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0),
			// Binding 1: Storage image
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1),
			// Binding 2: Uniform buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 2),
			// Binding 3: Vertex buffer 
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 3),
			// Binding 4: Index buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 4),
		};

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayout));

		std::vector<VkDescriptorSetLayoutBinding> materialSetLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1),
			//vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 2),
		};

		VkDescriptorSetLayoutCreateInfo materialSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(materialSetLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &materialSetLayoutCI, nullptr, &materialSetLayout));

		VkDescriptorSetLayout layouts[] = { descriptorSetLayout, materialSetLayout };

		VkPipelineLayoutCreateInfo pPipelineLayoutCI = vks::initializers::pipelineLayoutCreateInfo(layouts, 2);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCI, nullptr, &pipelineLayout));

		/*
			Setup ray tracing shader groups
		*/
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

		VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		uint32_t maxRecursion = 8;
		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(maxRecursion), &maxRecursion);

		// Ray generation group
		{
			shaderStages.push_back(loadShader(getShadersPath() + "raytracingreflections/raygen.rgen.spv", VK_SHADER_STAGE_RAYGEN_BIT_KHR));
			// Pass recursion depth for reflections to ray generation shader via specialization constant
			shaderStages.back().pSpecializationInfo = &specializationInfo;
			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			shaderGroup.generalShader = static_cast<uint32_t>(shaderStages.size()) - 1;
			shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
			shaderGroups.push_back(shaderGroup);
		}

		// Miss group
		{
			shaderStages.push_back(loadShader(getShadersPath() + "raytracingreflections/miss.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			shaderGroup.generalShader = static_cast<uint32_t>(shaderStages.size()) - 1;
			shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
			shaderGroups.push_back(shaderGroup);
		}

		// Shadow miss group
		{
			shaderStages.push_back(loadShader(getShadersPath() + "raytracingreflections/shadow.rmiss.spv", VK_SHADER_STAGE_MISS_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			shaderGroup.generalShader = static_cast<uint32_t>(shaderStages.size()) - 1;
			shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
			shaderGroups.push_back(shaderGroup);
		}

		// Closest hit group
		{
			shaderStages.push_back(loadShader(getShadersPath() + "raytracingreflections/closesthit.rchit.spv", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
			shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
			shaderGroup.generalShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.closestHitShader = static_cast<uint32_t>(shaderStages.size()) - 1;
			shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
			shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
			shaderGroups.push_back(shaderGroup);
		}

		VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCI = vks::initializers::rayTracingPipelineCreateInfoKHR();
		rayTracingPipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
		rayTracingPipelineCI.pStages = shaderStages.data();
		rayTracingPipelineCI.groupCount = static_cast<uint32_t>(shaderGroups.size());
		rayTracingPipelineCI.pGroups = shaderGroups.data();
		rayTracingPipelineCI.maxPipelineRayRecursionDepth = maxRecursion + 1;
		rayTracingPipelineCI.layout = pipelineLayout;
		VK_CHECK_RESULT(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rayTracingPipelineCI, nullptr, &pipeline));
	}

	/*
		Create the uniform buffer used to pass matrices to the ray tracing ray generation shader
	*/
	void createUniformBuffer()
	{
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&ubo,
			sizeof(uniformData),
			&uniformData));
		VK_CHECK_RESULT(ubo.map());

		updateUniformBuffers();
	}

	/*
		If the window has been resized, we need to recreate the storage image and it's descriptor
	*/
	void handleResize()
	{
		//m.lock();

		// Recreate image
		createStorageImage(swapChain.colorFormat, { width, height, 1 });
		// Update descriptor
		VkDescriptorImageInfo storageImageDescriptor{ VK_NULL_HANDLE, storageImage.view, VK_IMAGE_LAYOUT_GENERAL };
		VkWriteDescriptorSet resultImageWrite = vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, &storageImageDescriptor);
		vkUpdateDescriptorSets(device, 1, &resultImageWrite, 0, VK_NULL_HANDLE);
		resized = false;

		//m.unlock();
	}

	/*
		Command buffer generation
	*/
	void buildCommandBuffers()
	{
		if (resized)
		{
			handleResize();
		}

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		VkDescriptorSet sets[] = { descriptorSet, scene.materials[0].descriptorSet };

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 2, sets, 0, 0);

			/*
				Dispatch the ray tracing commands
			*/
			VkStridedDeviceAddressRegionKHR emptySbtEntry = {};
			vkCmdTraceRaysKHR(
				drawCmdBuffers[i],
				&shaderBindingTables.raygen.stridedDeviceAddressRegion,
				&shaderBindingTables.miss.stridedDeviceAddressRegion,
				&shaderBindingTables.hit.stridedDeviceAddressRegion,
				&emptySbtEntry,
				width,
				height,
				1);

			/*
				Copy ray tracing output to swap chain image
			*/

			// Prepare current swap chain image as transfer destination
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				swapChain.images[i],
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				subresourceRange);

			// Prepare ray tracing output image as transfer source
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				storageImage.image,
				VK_IMAGE_LAYOUT_GENERAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				subresourceRange);



			//vks::tools::insertImageMemoryBarrier(
			//	drawCmdBuffers[i],
			//	swapChain.images[i],
			//	0,
			//	VK_ACCESS_TRANSFER_WRITE_BIT,
			//	VK_IMAGE_LAYOUT_UNDEFINED,
			//	VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

			//// Transition swapchain image from present to transfer source layout
			//vks::tools::insertImageMemoryBarrier(
			//	drawCmdBuffers[i],
			//	storageImage.image,
			//	VK_ACCESS_MEMORY_READ_BIT,
			//	VK_ACCESS_TRANSFER_READ_BIT,
			//	VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			//	VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });




			VkImageCopy copyRegion{};
			copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.srcOffset = { 0, 0, 0 };
			copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
			copyRegion.dstOffset = { 0, 0, 0 };
			copyRegion.extent = { width, height, 1 };
			vkCmdCopyImage(drawCmdBuffers[i], storageImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapChain.images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

			// Transition swap chain image back for presentation
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				swapChain.images[i],
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				subresourceRange);

			// Transition ray tracing output image back to general layout
			vks::tools::setImageLayout(
				drawCmdBuffers[i],
				storageImage.image,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_IMAGE_LAYOUT_GENERAL,
				subresourceRange);

			//vks::tools::insertImageMemoryBarrier(
			//	drawCmdBuffers[i],
			//	swapChain.images[i],
			//	0,
			//	VK_ACCESS_TRANSFER_WRITE_BIT,
			//	VK_IMAGE_LAYOUT_UNDEFINED,
			//	VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });

			//// Transition swapchain image from present to transfer source layout
			//vks::tools::insertImageMemoryBarrier(
			//	drawCmdBuffers[i],
			//	storageImage.image,
			//	VK_ACCESS_MEMORY_READ_BIT,
			//	VK_ACCESS_TRANSFER_READ_BIT,
			//	VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			//	VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VK_PIPELINE_STAGE_TRANSFER_BIT,
			//	VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });


			drawUI(drawCmdBuffers[i], frameBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void updateUniformBuffers()
	{
		uniformData.projInverse = glm::inverse(camera.matrices.perspective);
		uniformData.viewInverse = glm::inverse(camera.matrices.view);

		// For rendering the normals correctly in the closest hit shader
		// This is only needed when doing moving objects where you have to 
		// recreate / reuse the top-level acceleration structure
		for (size_t i = 0; i < 3; i++)
			for (size_t j = 0; j < 4; j++)
				uniformData.transformation_matrix[j][i] = transformMatrix.matrix[i][j];


		const float speed = 0.125f;

		///		timer = 0;

		uniformData.light_positions[0] = glm::vec4(
			cos(glm::radians(timer * 360.0f)) * 40.0f * speed,
			-50.0f + sin(glm::radians(timer * 360.0f)) * 20.0f * speed,
			25.0f + sin(glm::radians(timer * 360.0f)) * 5.0f * speed,
			0.0f);

		//		uniformData.light_positions[0] = glm::vec4(0, -50, 0, 0);

		uniformData.light_positions[1] = uniformData.light_positions[0];
		uniformData.light_positions[1].x = -uniformData.light_positions[1].x;
		uniformData.light_positions[1].z = -uniformData.light_positions[1].z;

		uniformData.light_colors[0] = glm::vec4(1, 0, 0, 1);
		uniformData.light_colors[1] = glm::vec4(0, 0, 1, 1);

		uniformData.camera_pos = camera.position;

		// Pass the vertex size to the shader for unpacking vertices
		uniformData.vertexSize = sizeof(vkglTF::Vertex);

		memcpy(ubo.mapped, &uniformData, sizeof(uniformData));
	}

	void getEnabledFeatures()
	{
		// Enable features required for ray tracing using feature chaining via pNext		
		enabledBufferDeviceAddresFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
		enabledBufferDeviceAddresFeatures.bufferDeviceAddress = VK_TRUE;

		enabledRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
		enabledRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
		enabledRayTracingPipelineFeatures.pNext = &enabledBufferDeviceAddresFeatures;

		enabledAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
		enabledAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
		enabledAccelerationStructureFeatures.pNext = &enabledRayTracingPipelineFeatures;

		deviceCreatepNextChain = &enabledAccelerationStructureFeatures;
	}

	void prepare()
	{
		VulkanRaytracingSample::prepare();

		// Create the acceleration structures used to render the ray traced scene
		createBottomLevelAccelerationStructure();
		createTopLevelAccelerationStructure();

		createStorageImage(swapChain.colorFormat, { width, height, 1 });
		createUniformBuffer();
		createRayTracingPipeline();
		createShaderBindingTables();
		createDescriptorSets();
		buildCommandBuffers();
		prepared = true;
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VulkanExampleBase::submitFrame();
	}



	virtual void render()
	{
		if (!prepared)
			return;

		// To do: There's a way to update the structure instead of 
		// deleting and recreating it
		// see: https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/extensions/raytracing_extended
		// Note: this causes a bug which locks the app if window becomes non-minimized
		//deleteAccelerationStructure(topLevelAS);
		//createTopLevelAccelerationStructure();

		draw();


		if (!paused || camera.updated)
			updateUniformBuffers();

	}
};

VULKAN_EXAMPLE_MAIN()
