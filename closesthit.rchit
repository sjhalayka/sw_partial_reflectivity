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
};

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;
layout(location = 2) rayPayloadEXT bool shadowed;

layout(binding = 0, set = 1) uniform sampler2D baseColorSampler;
layout(binding = 1, set = 1) uniform sampler2D normalSampler;

hitAttributeEXT vec2 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 2, set = 0) uniform UBO 
{
	mat4 viewInverse;
	mat4 projInverse;
	vec4 lightPos;
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
vec3 phongModelDiffAndSpec(bool do_specular, float reflectivity, vec3 color, vec3 light_pos, vec3 frag_pos, vec3 frag_normal)
{
	const vec3 MaterialKs = vec3(1.0, 0.5, 0.0);
	const vec3 MaterialKa = vec3(0.0, 0.025, 0.075);
	const float MaterialShininess = 10.0;

	const vec3 n = normalize(frag_normal);
	const vec3 s = normalize(light_pos - frag_pos);
	const vec3 v = normalize(frag_pos);
	const vec3 r = reflect( -s, n );
	const float sDotN = max( dot(s,n), 0.0 ); // This second parameter affects the visibility of shadow edges
	const vec3 diffuse = color * sDotN;
	vec3 spec = vec3(0.0);

	if(sDotN > 0.0)
		spec = MaterialKs * pow( max( dot(r,v), 0.0 ), MaterialShininess );

	vec3 ret = diffuse + MaterialKa;

	if(do_specular)
		ret = ret + spec;//*reflectivity;
    
	return ret;
}



void main()
{
	ivec3 index = ivec3(indices.i[3 * gl_PrimitiveID], indices.i[3 * gl_PrimitiveID + 1], indices.i[3 * gl_PrimitiveID + 2]);

	Vertex v0 = unpack(index.x);
	Vertex v1 = unpack(index.y);
	Vertex v2 = unpack(index.z);

	// Interpolate
	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);
	vec3 pos = v0.pos * barycentricCoords.x + v1.pos * barycentricCoords.y + v2.pos * barycentricCoords.z;
	vec2 uv = v0.uv * barycentricCoords.x + v1.uv * barycentricCoords.y + v2.uv * barycentricCoords.z;

	// This will be a texture sample
	rayPayload.reflector = 0.5;//1.0;//length(texture(normalSampler, uv).rgb) / sqrt(3.0);
	rayPayload.opacity = 0.1;

	// This will be a texture sample
	vec3 color = texture(baseColorSampler, uv).rgb;//(v0.color.rgb + v1.color.rgb + v2.color.rgb) / 3.0;

	rayPayload.pure_color = color;
	rayPayload.color = phongModelDiffAndSpec(true, rayPayload.reflector, color, ubo.lightPos.xyz, pos, normal);// vec3(uv, 0.0);
	rayPayload.distance = gl_RayTmaxEXT;
	rayPayload.normal = normal;

	// Basic lighting
	vec3 lightVector = normalize(ubo.lightPos.xyz);

	if(dot(normal, lightVector) < 0.0)
	{
		shadowed = true;
	}
	else
	{
		// Shadow casting
		float tmin = 0.001;
		float tmax = 1000.0;

		vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
		vec3 biased_origin = origin + normal * 0.01;

		shadowed = true; // Make sure to set this to the default before tracing the ray!
		traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 0, 1, biased_origin, tmin, lightVector, tmax, 2);
	}

	if (shadowed)
	{
		rayPayload.color = phongModelDiffAndSpec(false, rayPayload.reflector, color, ubo.lightPos.xyz, pos, normal) * 0.3;
	}
}
