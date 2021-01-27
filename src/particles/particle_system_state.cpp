#include <cstdint>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <string_view>
#include <iostream>

#include <zlib.h>

#include <utils/io/compression.h>
#include <utils/io.h>

#include "particle_system_state.h"

using namespace std::literals;


namespace
{
	template <typename T>
	auto alloc_buffer(std::size_t num_particles)
	{
		return std::unique_ptr<T[]>(new T[num_particles]);
	}

	std::ostream& write_header(std::ostream& file, std::size_t num_particles, bool is_replay)
	{
		if (!write<std::int64_t>(file, is_replay ? -static_cast<std::int64_t>(num_particles) : static_cast<std::int64_t>(num_particles)))
			throw std::runtime_error("failed to write particles file header");

		return file;
	}

	auto read_header(std::istream& file)
	{
		auto num_particles = read<std::int64_t>(file);

		if (!file)
			throw std::runtime_error("failed to read particles file header");


		struct header_info
		{
			std::size_t num_particles;
			bool is_replay;
		};

		return header_info { static_cast<std::size_t>(num_particles < 0 ? -num_particles : num_particles), num_particles < 0 };
	}

	zlib_writer& write(zlib_writer& writer, const ParticleSystemParameters& params)
	{
		writer(params.bb_min);
		writer(params.bb_max);
		writer(params.min_particle_radius);
		writer(params.max_particle_radius);
		writer(params.gravity);
		//writer(params.damping);
		writer(params.bounce);
		writer(params.coll_attraction);
		writer(params.coll_damping);
		writer(params.coll_shear);
		writer(params.coll_spring);
		return writer;
	}

	zlib_reader& read(ParticleSystemParameters& params, zlib_reader& reader)
	{
		reader(params.bb_min);
		reader(params.bb_max);
		reader(params.min_particle_radius);
		reader(params.max_particle_radius);
		reader(params.gravity);
		//reader(params.damping);
		reader(params.bounce);
		reader(params.coll_attraction);
		reader(params.coll_damping);
		reader(params.coll_shear);
		reader(params.coll_spring);
		return reader;
	}

	zlib_writer& write_initial_particle_state(zlib_writer& writer, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	{
		writer(params);
		writer(&x[0], num_particles);
		writer(&y[0], num_particles);
		writer(&z[0], num_particles);
		writer(&r[0], num_particles);
		writer(&color[0], num_particles);

		return writer;
	}

	zlib_reader& read_initial_particle_state(zlib_reader& reader, std::size_t num_particles, float* x, float* y, float* z, float* r, std::uint32_t* color, ParticleSystemParameters& params)
	{
		reader(params);
		reader(&x[0], num_particles);
		reader(&y[0], num_particles);
		reader(&z[0], num_particles);
		reader(&r[0], num_particles);
		reader(&color[0], num_particles);

		return reader;
	}

	zlib_writer& write_frame_data(zlib_writer& writer, std::size_t num_particles, std::chrono::nanoseconds dt, const float* positions, const std::uint32_t* colors)
	{
		writer(static_cast<std::int64_t>(dt.count()));
		writer(positions, num_particles * 4);
		writer(colors, num_particles);

		return writer;
	}

	zlib_reader& read_frame_data(zlib_reader& reader, std::size_t num_particles, std::chrono::nanoseconds& dt, float* positions, std::uint32_t* colors)
	{
		dt = std::chrono::nanoseconds(reader.read<std::int64_t>());
		reader(positions, num_particles * 4);
		reader(colors, num_particles);

		return reader;
	}
}

std::ostream& save_particle_state(std::ostream& file, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
{
	write_header(file, num_particles, false);

	zlib_writer writer(file);
	write_initial_particle_state(writer, num_particles, x, y, z, r, color, params);

	return file;
}

ParticleReplayWriter::ParticleReplayWriter(std::ostream& file, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* color, const ParticleSystemParameters& params)
	: writer(write_header(file, num_particles, true)),
	  num_particles(num_particles)
{
	write_initial_particle_state(writer, num_particles, x, y, z, r, color, params);
}

void ParticleReplayWriter::add_frame(std::chrono::nanoseconds dt, const float* positions, const std::uint32_t* colors)
{
	write_frame_data(writer, num_particles, dt, positions, colors);
}

std::istream& load_particles(ParticleSystemBuilder& builder, std::istream& file)
{
	auto [num_particles, is_replay] = read_header(file);

	zlib_reader reader(file);

	ParticleSystemParameters params;
	auto position = alloc_buffer<float>(num_particles * 4);
	auto color = alloc_buffer<std::uint32_t>(num_particles);

	read_initial_particle_state(reader, num_particles, &position[0] + 0 * num_particles, &position[0] + 1 * num_particles, &position[0] + 2 * num_particles, &position[0] + 3 * num_particles, &color[0], params);

	if (is_replay)
	{
		auto& replay_builder = builder.add_particle_replay(num_particles, &position[0], &color[0], params);

		while (file)
		{
			std::chrono::nanoseconds dt;
			read_frame_data(reader, num_particles, dt, &position[0], &color[0]);
			replay_builder.add_frame(dt, &position[0], &color[0]);
		}
	}
	else
	{
		builder.add_particle_simulation(num_particles, std::move(position), std::move(color), params);
	}

	return file;
}


void save_particle_state(const std::filesystem::path& filename, std::size_t num_particles, const float* x, const float* y, const float* z, const float* r, const std::uint32_t* colors, const ParticleSystemParameters& params)
{
	std::ofstream file(filename, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open '" + filename.string() + '\'');

	save_particle_state(file, num_particles, x, y, z, r, colors, params);
}

void load_particles(ParticleSystemBuilder& builder, const std::filesystem::path& filename)
{
	std::ifstream file(filename, std::ios::binary);

	if (!file)
		throw std::runtime_error("failed to open '" + filename.string() + '\'');

	load_particles(builder, file);
}
