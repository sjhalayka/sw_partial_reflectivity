#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable


struct RayPayload {
	vec3 color;
	vec3 pure_color;
	float distance;
	vec3 normal;
	float reflector;
	float opacity;
	vec3 pos;
	vec3 wro;
	vec3 wrd;
	float hitt;
	double depth;
};

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;

layout(location = 2) rayPayloadEXT bool shadowed;

layout(binding = 0, set = 1) uniform sampler2D baseColorSampler;
layout(binding = 1, set = 1) uniform sampler2D normalSampler;
layout(binding = 2, set = 1) uniform sampler2D occlusionSampler;

hitAttributeEXT vec2 attribs;




const int max_lights = 2;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, set = 0) uniform UBO 
{
	mat4 viewInverse;
	mat4 projInverse;

	mat4 transformation_matrix;

	vec4 light_positions[max_lights];
	vec4 light_colors[max_lights];

	vec3 camera_pos;
	int vertexSize;
} ubo;
layout(binding = 3, set = 0) buffer Vertices { vec4 v[]; } vertices;
layout(binding = 4, set = 0) buffer Indices { uint i[]; } indices;



struct Vertex
{
  vec3 pos;
  vec3 normal;
  vec2 uv;
  vec4 color;
  vec4 _pad0; 
  vec4 _pad1;
};

Vertex unpack(uint index)
{
	// Unpack the vertices from the SSBO using the glTF vertex structure
	// The multiplier is the size of the vertex divided by four float components (=16 bytes)
	const int m = ubo.vertexSize / 16;

	vec4 d0 = vertices.v[m * index + 0];
	vec4 d1 = vertices.v[m * index + 1];
	vec4 d2 = vertices.v[m * index + 2];

	Vertex v;
	v.pos = d0.xyz;
	v.normal = vec3(d0.w, d1.x, d1.y);
	v.uv = vec2(d1.z, d1.w);
	v.color = vec4(d2.x, d2.y, d2.z, 1.0);

	return v;
}



// https://github.com/daw42/glslcookbook/blob/master/chapter07/shader/shadowmap.fs
vec3 phongModelDiffAndSpec(bool do_specular, float reflectivity, vec3 color, vec3 light_color, vec3 light_pos, vec3 frag_pos, vec3 frag_normal)
{
	const vec3 MaterialKs = vec3(1.0, 0.5, 0.0);
	const vec3 MaterialKa = vec3(0.0, 0.025, 0.075);
	const float MaterialShininess = 10.0;

	const vec3 n = normalize(frag_normal);
	const vec3 s = normalize(light_pos - frag_pos);
	const vec3 v = normalize(frag_pos);
	const vec3 r = reflect( -s, n );
	const float sDotN = max( dot(s,n), 0.0 ); // This second parameter affects the visibility of shadow edges
	const vec3 diffuse = light_color * color * sDotN;
	vec3 spec = vec3(0.0);

	if(sDotN > 0.0)
		spec = MaterialKs * pow( max( dot(r,v), 0.0 ), MaterialShininess );

	vec3 ret = diffuse + MaterialKa;

	if(do_specular)
		ret = ret + spec;
    
	return ret;

}



uint prng_state = 0;

// See: https://github.com/nvpro-samples/vk_mini_path_tracer/blob/main/vk_mini_path_tracer/shaders/raytrace.comp.glsl#L26
float stepAndOutputRNGFloat(inout uint rngState)
{
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rngState  = rngState * 747796405 + 1;
  uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}







float get_caustic_float(const vec3 light_pos, const vec3 normal, float caustic_sharpness)
{
	// Back up the payload before we trace some rays
	RayPayload r = rayPayload;

	vec3 lightVector = normalize(light_pos);

	// Pseudorandomize the direction of the light
	// in order to get blurry caustics
	vec3 rdir = normalize(vec3(stepAndOutputRNGFloat(prng_state), stepAndOutputRNGFloat(prng_state), stepAndOutputRNGFloat(prng_state)));

	// Stick to the correct hemisphere
	if(dot(rdir, lightVector) < 0.0)
		rdir = -rdir;
	
	// Keep the caustics stay dynamic to some degree
	// I mean, how blurry do you really need the edges to be?
	if(caustic_sharpness < 0.5)
		caustic_sharpness = 0.5;


	lightVector = mix(rdir, lightVector, caustic_sharpness);

	float caustic = 0.0;

	vec3 first_origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 first_biased_origin = first_origin + normal * 0.01;

	vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 biased_origin = origin + normal * 0.01;

	if(dot(normal, lightVector) < 0.0)
	{
		caustic = 0.0;
	}
	else
	{
		// caustic casting
		origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
		biased_origin = origin + normal * 0.01;

		caustic = 0.0;

		bool first_assignment = true;

		while(true)
		{
			rayPayload.depth = 1;

			traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, biased_origin, 0.001, lightVector, 10000.0, 0);

			if(rayPayload.distance == -1)
				break;

			biased_origin = biased_origin + lightVector*0.01;				

			float first_dot = dot(normalize(lightVector), normalize(rayPayload.normal));
			float first_opacity = rayPayload.opacity;

			float rating = abs(first_dot);

			rating = mix(rating, 0, first_opacity);

			if(first_assignment || rating > caustic)
			{
				caustic = rating;
				first_assignment = false;
			}
		}
	}

	// Restore the payload after we've traced some rays
	rayPayload = r;

	return caustic;//clamp(caustic, 0.0, 1.0);
}








float get_shadow_float(const vec3 light_pos, const vec3 normal, float shadow_sharpness)
{
	// Back up the payload before we trace some rays
	RayPayload r = rayPayload;

	vec3 lightVector = normalize(light_pos);

	// Pseudorandomize the direction of the light
	// in order to get blurry shadows
	vec3 rdir = normalize(vec3(stepAndOutputRNGFloat(prng_state), stepAndOutputRNGFloat(prng_state), stepAndOutputRNGFloat(prng_state)));

	// Stick to the correct hemisphere
	if(dot(rdir, lightVector) < 0.0)
		rdir = -rdir;
	
	// Keep the shadows stay dynamic to some degree
	// I mean, how blurry do you really need the edges to be?
	if(shadow_sharpness < 0.5)
		shadow_sharpness = 0.5;

	lightVector = mix(rdir, lightVector, shadow_sharpness);

	float shadow = 0.0;

	vec3 first_origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 first_biased_origin = first_origin + normal * 0.01;

	vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 biased_origin = origin + normal * 0.01;

	if(dot(normal, lightVector) < 0.0)
	{
		shadow = 0.0;
	}
	else
	{
		// Shadow casting
		origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
		biased_origin = origin + normal * 0.01;

		shadow = 0.0;

		bool first_assignment = true;

		while(true)
		{
			rayPayload.depth = 1;

			traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, biased_origin, 0.001, lightVector, 10000.0, 0);

			if(rayPayload.distance == -1)
				break;

			biased_origin = biased_origin + lightVector*0.01;				

			float first_dot = dot(normalize(lightVector), normalize(rayPayload.normal));
			float first_opacity = rayPayload.opacity;

			float rating = 1.0 - abs(first_dot);

			rating = mix(rating, 1, first_opacity);

			if(first_assignment || rating < shadow)
			{
				shadow = rating;
				first_assignment = false;
			}
		}

		shadow = 1 - shadow;
	}

	// Restore the payload after we've traced some rays
	rayPayload = r;

	return shadow;//clamp(shadow, 0.0, 1.0);
}






void main()
{
	const ivec2 pixel_pos = ivec2(gl_LaunchIDEXT.xy);
	const ivec2 res = ivec2(gl_LaunchSizeEXT.xy);
	prng_state = res.x * pixel_pos.y + pixel_pos.x; // Each pixel gets its own unique seed

	ivec3 index = ivec3(indices.i[3 * gl_PrimitiveID], indices.i[3 * gl_PrimitiveID + 1], indices.i[3 * gl_PrimitiveID + 2]);

	Vertex v0 = unpack(index.x);
	Vertex v1 = unpack(index.y);
	Vertex v2 = unpack(index.z);

	// Interpolate
	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);
	const vec3 pos = v0.pos * barycentricCoords.x + v1.pos * barycentricCoords.y + v2.pos * barycentricCoords.z;
	const vec2 uv = v0.uv * barycentricCoords.x + v1.uv * barycentricCoords.y + v2.uv * barycentricCoords.z;

	vec4 n = ubo.transformation_matrix*vec4(normal, 0.0);
	
	rayPayload.reflector = 0.75;
	rayPayload.opacity = pow(length(texture(normalSampler, uv).rgb) / sqrt(3.0), 1.0);

	vec3 color = texture(baseColorSampler, uv).rgb;

	rayPayload.pure_color = color;
	rayPayload.distance = gl_RayTmaxEXT;
	rayPayload.normal = n.xyz;
	
	rayPayload.wro = gl_WorldRayOriginEXT;
	rayPayload.wrd = gl_WorldRayDirectionEXT;
	rayPayload.hitt = gl_HitTEXT;

	rayPayload.color = vec3(0, 0, 0);
	
	if(rayPayload.depth != 1)
	{
		for (int i = 0; i < max_lights; i++)
		{
			float s = get_shadow_float(ubo.light_positions[i].xyz, rayPayload.normal, rayPayload.reflector);	
			float c = get_caustic_float(ubo.light_positions[i].xyz, rayPayload.normal, rayPayload.reflector);

			rayPayload.color += s*phongModelDiffAndSpec(true, rayPayload.reflector, color, ubo.light_colors[i].rgb, ubo.light_positions[i].xyz, pos, rayPayload.normal);
			rayPayload.color += c*ubo.light_colors[i].rgb;
		}
	}
}
