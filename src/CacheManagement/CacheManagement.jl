module CacheManagement

using ..KSTypes
using Base.Threads


# Functions - Cache Operations
export cache_size, cache_stats
export clear_cache!, clear_cache_entry!, clear_invalid_cache!, clear_transform_cache!
export get_access_count, get_or_create_cached_item
export get_or_create_standard_cell, get_or_create_standard_spectral_property
export is_valid_cached_cell
export maybe_cleanup_cache!
export thread_safe_cache_operation
export update_access_count!
export update_variable_cache!
export clear_variable_cache!
# Constants
export CACHE_LOCK
export STANDARD_CELLS, STANDARD_CELLS_LOCK
export STANDARD_SPECTRAL_PROPERTIES, STANDARD_SPECTRAL_PROPERTIES_LOCK
export TRANSFORM_CACHE, TRANSFORM_CACHE_LOCKS

const CACHE_LOCK = ReentrantLock()
const TRANSFORM_CACHE_LOCKS = Dict{Symbol, ReentrantLock}()
const TRANSFORM_CACHE = KSTypes.TransformCache(
	max_size = 1000, cleanup_threshold = 0.8)
const STANDARD_CELLS = Dict{Any, KSTypes.KSCell}()
const STANDARD_SPECTRAL_PROPERTIES = Dict{
	Tuple{Int, Int}, KSTypes.StandardSpectralProperties}()
const STANDARD_CELLS_LOCK = ReentrantLock()
const STANDARD_SPECTRAL_PROPERTIES_LOCK = ReentrantLock()

# Function to clear the transform cache

"""
	thread_safe_cache_operation(f::Function)

Execute a function with thread-safe cache access.
"""
function thread_safe_cache_operation(f::Function)
	lock(CACHE_LOCK)
	try
		return f()
	finally
		unlock(CACHE_LOCK)
	end
end

"""
	cache_size()

Get the current size of the transform cache.
"""
function cache_size()
	lock(TRANSFORM_CACHE._lock) do
		return length(TRANSFORM_CACHE.cells)
	end
end

"""
    clear_transform_cache!()

Clear all cached transforms and related data completely.
Thread-safe implementation respecting type parameters.
"""
function clear_transform_cache!()
    local cache = CacheManagement.TRANSFORM_CACHE

    # Use a single lock for the entire operation
    lock(CacheManagement.CACHE_LOCK) do
        # Clear dictionaries with proper reference cleanup
        for (_, cell_lock) in cache.cell_locks
            # Ensure any pending operations are complete
            lock(cell_lock) do
                # No-op, just ensuring lock is acquired
            end
        end

        # Clear all collections atomically
        cache.cells = Dict{Any, Any}()
        cache.operator_cache = Dict{Any, Any}()
        cache.matrix_cache = Dict{Any, Any}()
        cache.cell_locks = Dict{Any, ReentrantLock}()

        # Reset stats while maintaining type stability
        cache.stats = CacheStats()
        cache.stats.custom = Dict{Any, Int}()

        # Force immediate garbage collection
        GC.gc(true)
    end

    return nothing
end
"""
	get_or_create_cached_item(f::Function, cache::CacheManager{T}, key) where T

Retrieve or create a cached item using thread-safe access.
"""
function get_or_create_cached_item(f::Function, cache::CacheManager{T}, key) where T
	lock(cache.lock) do
		if haskey(cache.items, key)
			update_access!(cache, key)
			return cache.items[key]
		end

		if length(cache.items) >= cache.capacity
			evict_item!(cache)
		end

		item = f()
		store_item!(cache, key, item)
		update_access!(cache, key)  # Count first access after storing
		return item
	end
end

# Do-block syntax support
function get_or_create_cached_item(cache::CacheManager{T}, key, f::Function) where T
	return get_or_create_cached_item(f, cache, key)
end

"""
	clear_cache!(cache::CacheManager)

Remove all items from the cache.
"""
function clear_cache!(cache::CacheManager)
	lock(cache.lock) do
		empty!(cache.items)
		empty!(cache.metadata)
		if cache.strategy isa FIFOEviction
			empty!(cache.strategy.insertion_order)
		end
	end
end

"""
	clear_cache_entry!(cache::CacheManager, key)

Remove a specific item from the cache.
"""
function clear_cache_entry!(cache::CacheManager, key)
	lock(cache.lock) do
		if haskey(cache.items, key)
			remove_item!(cache, key)
		end
	end
end

"""
	cache_stats(cache::CacheManager)

Get cache statistics including hit/miss counts.
"""
function cache_stats(cache::CacheManager)
	lock(cache.lock) do
		total_access = sum(meta.access_count for meta in values(cache.metadata))
		total_size = sum(meta.size for meta in values(cache.metadata))
		return (
			items = length(cache.items),
			total_access = total_access,
			total_size = total_size,
		)
	end
end

# Helper functions for cache operations
function update_access!(cache::CacheManager, key)
	meta = cache.metadata[key]
	meta.access_count += 1
	return meta.last_accessed = time()
end

function store_item!(cache::CacheManager, key, item)
	meta = CacheMetadata(0, time(), time(), sizeof(item))  # Initialize access_count to 0
	cache.metadata[key] = meta
	cache.items[key] = item

	# Track insertion order for FIFO
	if cache.strategy isa FIFOEviction
		push!(cache.strategy.insertion_order, key)
	end

	if !isnothing(cache.on_insert)
		cache.on_insert(key, item, meta)
	end
end

function evict_item!(cache::CacheManager{T, <:LRUEviction}) where T
	isempty(cache.metadata) && return nothing
	oldest_access = Inf
	oldest_key = first(keys(cache.metadata))

	for (key, meta) in cache.metadata
		if meta.last_accessed < oldest_access
			oldest_access = meta.last_accessed
			oldest_key = key
		end
	end

	if !isnothing(oldest_key)
		remove_item!(cache, oldest_key)
	end
end
function evict_item!(cache::CacheManager{T, <:CustomEviction}) where T
	key_to_evict = cache.strategy.evict(cache.items, cache.metadata)
	if !isnothing(key_to_evict)
		remove_item!(cache, key_to_evict)
	end
end

function evict_item!(cache::CacheManager{T, <:FIFOEviction}) where T
	isempty(cache.metadata) && return nothing

	# Get the first inserted item from our tracked order
	while !isempty(cache.strategy.insertion_order)
		key_to_evict = popfirst!(cache.strategy.insertion_order)
		if haskey(cache.items, key_to_evict)
			remove_item!(cache, key_to_evict)
			return nothing
		end
	end
end

function remove_item!(cache::CacheManager, key)
	if !isnothing(cache.on_evict)
		cache.on_evict(key, cache.items[key], cache.metadata[key])
	end

	delete!(cache.items, key)
	return delete!(cache.metadata, key)
end

function clear_invalid_cache!(cache::TransformCache)
	lock(cache._lock) do
		for (key, cell) in cache.cells
			if !is_valid_cached_cell(cell)
				delete!(cache.cells, key)
				delete!(cache.cell_locks, key)
				cache.stats.evictions += 1
			end
		end
	end
end

# Add adaptive cleanup
function maybe_cleanup_cache!(cache::TransformCache)
	if length(cache.cells) > cache.max_size
		lock(cache._lock) do
			# Sort by access count in descending order
			sorted_items = sort(collect(pairs(cache.cells));
				by = k -> get(cache.stats.custom, k.first, 0),
				rev = true)

			# Keep only max_size most accessed items
			to_remove = sorted_items[(cache.max_size + 1):end]

			# Remove excess items
			for (key, _) in to_remove
				delete!(cache.cells, key)
				delete!(cache.cell_locks, key)
				delete!(cache.stats.custom, key)
				cache.stats.evictions += 1
			end
		end
	end
end

function is_valid_cached_cell(cell::KSCell)
	return cell.is_leaf && !cell.is_fictitious
end

function get_access_count(item_pair::Tuple)
	key, _ = item_pair
	lock(TRANSFORM_CACHE._lock) do
		return get(TRANSFORM_CACHE.stats.custom, key, 0)
	end
end

function update_access_count!(key::Any)
	lock(TRANSFORM_CACHE._lock) do
		count = get(TRANSFORM_CACHE.stats.custom, key, 0)
		TRANSFORM_CACHE.stats.custom[key] = count + 1
		TRANSFORM_CACHE.stats.hits += 1
	end
end

function clear_variable_cache!(cell::KSCell)
    empty!(cell.cache)
    for var_data in values(cell.variable_data)
        empty!(var_data)
    end
end

function update_variable_cache!(
    cell::KSCell{T,N},
    variable::Int,
    key::Symbol,
    value::Any) where {T,N}

    var_data = get!(Dict{Symbol,Any}, cell.variable_data, variable)
    var_data[key] = value
end



end # module CacheManagement
