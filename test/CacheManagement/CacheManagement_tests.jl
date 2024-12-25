using Test
using KitchenSink.KSTypes
using KitchenSink.CacheManagement
# include("../test_utils.jl")

# Test setup helper functions
function create_test_cell_1D(id::Int, p::Int)
	return KSCell{Float64, 1}(
		id,                     # id
		(p,),                  # p
		1,                     # level
		(1,),                  # continuity_order
		((p,), 1),            # standard_cell_key
		Dict{Symbol, Int}(:dim1_neg => id - 1, :dim1_pos => id + 1),  # neighbors
		Dict{NTuple{1, Int}, Int}((1,) => id),  # node_map
		(fill(true, p),),      # tensor_product_mask
		Dict{Symbol, Int}(:dim1_neg => id - 1, :dim1_pos => id + 1),  # boundary_connectivity
		0.1,                   # error_estimate
		0.05,                  # legendre_decay_rate
		true,                  # is_leaf
		false,                 # is_fictitious
		nothing,               # refinement_options
		nothing,               # parent_id
		nothing,                # child_ids
	)
end

function create_test_cell_2D(id::Int, p1::Int, p2::Int)
	return KSCell{Float64, 2}(
		id,                     # id
		(p1, p2),              # p
		1,                     # level
		(1, 1),                # continuity_order
		((p1, p2), 1),         # standard_cell_key
		Dict{Symbol, Int}(:dim1_neg => id - 1, :dim1_pos => id + 1,
			:dim2_neg => -1, :dim2_pos => -1),  # neighbors
		Dict{NTuple{2, Int}, Int}((1, 1) => id),  # node_map
		(fill(true, p1), fill(true, p2)),  # tensor_product_mask
		Dict{Symbol, Int}(:dim1_neg => id - 1, :dim1_pos => id + 1,
			:dim2_neg => -1, :dim2_pos => -1),  # boundary_connectivity
		0.1,                   # error_estimate
		0.05,                  # legendre_decay_rate
		true,                  # is_leaf
		false,                 # is_fictitious
		nothing,               # refinement_options
		nothing,               # parent_id
		nothing,                # child_ids
	)
end

@testset "CacheManagement Tests" begin
	@testset "Cache Type Tests" begin
		# 1. CacheStats structure tests
		@testset "CacheStats" begin
			stats = CacheStats()
			@test stats.hits == 0
			@test stats.misses == 0
			@test stats.evictions == 0
			@test stats.custom isa Dict
		end

		# 2. CacheMetadata structure tests
		@testset "CacheMetadata" begin
			meta = CacheMetadata(0, time(), time(), 100)
			@test meta.access_count == 0
			@test meta.size == 100
			@test meta.custom isa Dict{Symbol, Any}
		end

		# 3. Cache eviction strategy tests
		@testset "Cache Eviction Strategies" begin
			@testset "LRUEviction" begin
				strategy = LRUEviction()
				@test strategy isa CacheEvictionStrategy
			end

			@testset "FIFOEviction" begin
				strategy = FIFOEviction()
				@test strategy isa CacheEvictionStrategy
				@test isempty(strategy.insertion_order)
			end

			@testset "CustomEviction" begin
				evict_fn = (items, meta) -> first(keys(items))
				strategy = CustomEviction(evict_fn)
				@test strategy.evict === evict_fn
			end
		end

		# 4. CacheManager structure tests
		@testset "CacheManager Structure" begin
			cache = CacheManager{Int}(5)
			@test cache.capacity == 5
			@test isempty(cache.items)
			@test isempty(cache.metadata)
			@test cache.strategy isa LRUEviction
		end

		# 5. TransformCache structure tests
		@testset "TransformCache Structure" begin
			cache = TransformCache{Float64}()
			@test isempty(cache.cells)
			@test isempty(cache.operator_cache)
			@test isempty(cache.matrix_cache)
			@test cache.stats isa CacheStats
		end
	end

	@testset "CacheManager Basic Operations" begin
		cache = CacheManager{KSCell{Float64, 1}}(5)

		# Test initial state
		@test isempty(cache.items)
		@test isempty(cache.metadata)

		# Test item creation and retrieval
		test_cell = create_test_cell_1D(1, 3)
		key = ((3,), 1)

		retrieved_cell = get_or_create_cached_item(cache, key) do
			test_cell
		end

		@test retrieved_cell == test_cell
		@test haskey(cache.items, key)
		@test haskey(cache.metadata, key)

		# Test metadata
		meta = cache.metadata[key]
		@test meta.access_count == 1
		@test meta.size > 0
		@test meta.last_accessed ≤ time()
		@test meta.created ≤ time()
	end

	@testset "Cache Eviction Strategies" begin
		@testset "LRU Eviction" begin
			cache = CacheManager{Int}(3; strategy = :lru)

			# Fill cache
			for i in 1:3
				get_or_create_cached_item(cache, i) do
					i
				end
			end

			# Access item 1 to make it most recently used
			get_or_create_cached_item(cache, 1) do
				1
			end

			# Add new item to trigger eviction
			get_or_create_cached_item(cache, 4) do
				4
			end

			# Item 2 should be evicted (least recently used)
			@test !haskey(cache.items, 2)
			@test haskey(cache.items, 1)
			@test haskey(cache.items, 3)
			@test haskey(cache.items, 4)
		end

		@testset "FIFO Eviction" begin
			cache = CacheManager{Int}(3; strategy = :fifo)

			# Fill cache
			for i in 1:3
				get_or_create_cached_item(cache, i) do
					i
				end
			end

			# Access item 1 (shouldn't affect FIFO order)
			get_or_create_cached_item(cache, 1) do
				1
			end

			# Add new item to trigger eviction
			get_or_create_cached_item(cache, 4) do
				4
			end

			# Item 1 should be evicted (first in)
			@test !haskey(cache.items, 1)
			@test haskey(cache.items, 2)
			@test haskey(cache.items, 3)
			@test haskey(cache.items, 4)
		end
	end

	@testset "Thread Safety" begin
		cache = CacheManager{Int}(10)
		n_threads = 4
		n_iterations = 100

		# Test concurrent access
		results = Vector{Bool}(undef, n_threads * n_iterations)

		Threads.@threads for i in 1:n_threads
			for j in 1:n_iterations
				idx = (i - 1) * n_iterations + j
				try
					get_or_create_cached_item(cache, idx) do
						sleep(0.001)  # Simulate work
						idx
					end
					results[idx] = true
				catch
					results[idx] = false
				end
			end
		end

		@test all(results)
		@test length(cache.items) ≤ cache.capacity
	end

	@testset "Custom Callbacks" begin
		evicted_items = Dict()
		inserted_items = Dict()

		cache = CacheManager{Int}(2;
			on_evict = (key, item, meta) -> evicted_items[key] = item,
			on_insert = (key, item, meta) -> inserted_items[key] = item,
		)

		# Test insertion callback
		get_or_create_cached_item(cache, :a) do
			1
		end
		@test haskey(inserted_items, :a)
		@test inserted_items[:a] == 1

		# Test eviction callback
		get_or_create_cached_item(cache, :b) do
			2
		end
		get_or_create_cached_item(cache, :c) do
			3
		end  # Should trigger eviction of :a

		@test haskey(evicted_items, :a)
		@test evicted_items[:a] == 1
	end

	@testset "TransformCache" begin
		clear_transform_cache!()

		@testset "Basic Operations" begin
			# Test initial state
			@test cache_size() == 0

			# Test cell storage and retrieval
			test_cell = create_test_cell_1D(1, 3)

			lock(TRANSFORM_CACHE._lock) do
				TRANSFORM_CACHE.cells[:test_key] = test_cell
				@test haskey(TRANSFORM_CACHE.cells, :test_key)
				@test TRANSFORM_CACHE.cells[:test_key] == test_cell
			end

			# Test clearing cache
			clear_transform_cache!()
			@test cache_size() == 0
		end

		@testset "Thread-safe Operations" begin
			result = thread_safe_cache_operation() do
				create_test_cell_1D(1, 3)
			end

			@test result isa KSCell{Float64, 1}
		end
	end

	@testset "Cache Statistics" begin
		cache = CacheManager{Int}(5)

		# Create some items
		for i in 1:3
			get_or_create_cached_item(cache, i) do
				i
			end
		end

		# Access some items multiple times
		for _ in 1:5
			get_or_create_cached_item(cache, 1) do
				1
			end
		end

		stats = cache_stats(cache)
		@test stats.items == 3
		@test stats.total_access == 8
		@test stats.total_size > 0
	end

	@testset "Error Handling" begin
		cache = CacheManager{Int}(3)

		@test_throws ArgumentError CacheManager{Int}(0)
		@test_throws ArgumentError CacheManager{Int}(-1)

		# Test eviction on empty cache
		empty!(cache.items)
		empty!(cache.metadata)
		@test_nowarn get_or_create_cached_item(cache, :new) do
			1
		end

		# Test concurrent writes
		cache = CacheManager{Int}(2)
		@sync begin
			for i in 1:4
				Threads.@spawn begin
					get_or_create_cached_item(cache, i) do
						i
					end
				end
			end
		end
		@test length(cache.items) <= cache.capacity
	end
	@testset "Enhanced Cache Operations" begin
		@testset "Cache Validation and Cleanup" begin
			cache = TransformCache(; max_size = 10, cleanup_threshold = 0.8)

			# Create test cells
			test_cells = [create_test_cell_1D(i, 3) for i in 1:15]

			# Fill cache beyond threshold
			for (i, cell) in enumerate(test_cells)
				lock(cache._lock) do
					cache.cells[i] = cell
					cache.cell_locks[i] = ReentrantLock()
				end
			end

			# Test cleanup threshold trigger
			@test length(cache.cells) > cache.max_size * cache.cleanup_threshold
			maybe_cleanup_cache!(cache)
			@test length(cache.cells) <= cache.max_size

			# Verify eviction stats
			@test cache.stats.evictions > 0
		end

		@testset "Invalid Cache Cleanup" begin
			cache = TransformCache()

			# Create mix of valid and invalid cells
			valid_cell = create_test_cell_1D(1, 3)
			invalid_cell = create_test_cell_1D(2, 3)
			invalid_cell.is_leaf = false  # Make invalid

			lock(cache._lock) do
				cache.cells[:valid] = valid_cell
				cache.cells[:invalid] = invalid_cell
				cache.cell_locks[:valid] = ReentrantLock()
				cache.cell_locks[:invalid] = ReentrantLock()
			end

			clear_invalid_cache!(cache)

			@test haskey(cache.cells, :valid)
			@test !haskey(cache.cells, :invalid)
			@test cache.stats.evictions == 1
		end

		@testset "Thread Safety for Cleanup Operations" begin
			cache = TransformCache(; max_size = 5)
			n_threads = 4
			n_operations = 20

			# Concurrent cache operations
			@sync begin
				for t in 1:n_threads
					Threads.@spawn begin
						for i in 1:n_operations
							if rand() > 0.5
								maybe_cleanup_cache!(cache)
							else
								clear_invalid_cache!(cache)
							end
						end
					end
				end
			end

			# Verify cache integrity
			@test length(cache.cells) <= cache.max_size
			@test length(cache.cells) == length(cache.cell_locks)
		end

		@testset "Access Count Tracking" begin
			cache = TransformCache()
			test_cell = create_test_cell_1D(1, 3)

			lock(cache._lock) do
				cache.cells[:test] = test_cell
				cache.cell_locks[:test] = ReentrantLock()
			end

			# Simulate multiple accesses
			n_accesses = 5
			for _ in 1:n_accesses
				lock(cache._lock) do
					cache.stats.hits += 1
				end
			end

			# Test access count reflection
			@test cache.stats.hits == n_accesses
			@test cache.stats.misses == 0
		end

		@testset "Cache Size Management" begin
			clear_transform_cache!()
			@test isempty(TRANSFORM_CACHE.cells)
			@test isempty(TRANSFORM_CACHE.cell_locks)

			cache = TRANSFORM_CACHE
			cache.max_size = 3
			cache.cleanup_threshold = 0.7

			# Add items up to threshold
			for i in 1:4
				lock(cache._lock) do
					cache.cells[i] = create_test_cell_1D(i, 3)
					cache.cell_locks[i] = ReentrantLock()
				end
			end

			# Verify automatic cleanup
			maybe_cleanup_cache!(cache)
			@test length(cache.cells) <= cache.max_size
			@test cache.stats.evictions > 0

			@testset "Clear Transform Cache" begin
				clear_transform_cache!()
				@test isempty(TRANSFORM_CACHE.cells)
				@test isempty(TRANSFORM_CACHE.cell_locks)
				@test isempty(TRANSFORM_CACHE.stats.custom)
			end
		end
	end
	@testset "Item-specific Access Counts" begin
		cache = TransformCache()
		test_key = :test_item

		# Test initial count
		@test get_access_count((test_key, nothing)) == 0

		# Test incremental counts
		lock(cache._lock) do
			cache.cells[test_key] = create_test_cell_1D(1, 3)
			update_access_count!(test_key)
			@test get_access_count((test_key, nothing)) == 1

			update_access_count!(test_key)
			@test get_access_count((test_key, nothing)) == 2
		end
	end
end
