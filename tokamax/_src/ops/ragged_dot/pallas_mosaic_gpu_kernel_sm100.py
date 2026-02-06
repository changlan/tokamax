# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ragged dot Pallas-Mosaic-GPU Non-Quantized Kernel (Blackwell)."""
import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer  # pylint: disable=g-multiple-import,g-importing-member
from tokamax._src import jaxtyping
from tokamax._src.ops.ragged_dot import base
from tokamax._src.ops.ragged_dot import pallas_mosaic_gpu_common as common

_COMPUTE_WG = 0
_MMA_WARP = 0
_TMA_WARP = 1
_STORE_WG = 1

_TCGEN05_TRANSPOSED = plgpu.Layout.TCGEN05_TRANSPOSED
_TMEM = plgpu.Layout.TCGEN05_TMEM_NATIVE


@jaxtyping.jaxtyped
def ragged_dot_gpu_non_quant_blackwell_kernel(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "G K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "M N"]:
  """Pallas kernel for ragged dot with GPU quantization."""
  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  num_stages = config.num_stages
  collective = config.collective
  # `tile` is for each block
  tile_m = block_m
  tile_n = block_n
  if collective:
    block_m *= 2
    block_n *= 2

  w, x = (rhs.mT, lhs)

  (num_groups, n, k_w), (m, k_x) = w.shape, x.shape
  if k_w != k_x:
    raise ValueError(
        f"Contraction dim mismatch: weights.shape[1]={k_w}, x.shape[-1]={k_x}"
    )
  if group_sizes.shape != (num_groups,):
    raise ValueError(
        "Expected group_sizes to have shape"
        f" {(num_groups,)} but got {group_sizes.shape}"
    )
  if (x.dtype, w.dtype) != (jnp.bfloat16, jnp.bfloat16):
    raise ValueError(
        "Only the same precision bfloat16 x bfloat16 supported, got:"
        f" {x.dtype=} {w.dtype=}."
    )

  # num_stages must be less than or equal to the number of blocks
  num_stages = min(num_stages, k_w // block_k)

  group_info = common.GroupInfo.create_aligned(
      group_sizes, block_m, pl.cdiv(m, block_m) + num_groups - 1
  )
  m_iters = pl.cdiv(m, block_m) + num_groups - 1
  n_iters = pl.cdiv(n, block_n)

  def kernel(*refs, scoped):
    (
        x_gmem,
        w_gmem,
        _,
        group_id_gmem,
        start_within_block_gmem,
        actual_size_gmem,
        block_start_gmem,
        out_gmem,
    ) = refs
    scratch_buffers, barriers = scoped
    x_smem, w_smem, acc_smem, acc_tmem = scratch_buffers
    (
        xw_tma_barrier,
        consumed_barrier,
        mma_done_barrier,
        store_done_barrier,
    ) = barriers

    m, k = x_gmem.shape
    num_k_iters = pl.cdiv(k, block_k)
    cluster_idx = lax.axis_index("x")

    @plgpu.nd_loop((m_iters * n_iters,), collective_axes=("sm",), init_carry=0)
    def mn_loop(loop_info: plgpu.NDLoopInfo, carry):
      (lin_idx,) = loop_info.index
      tid_m, ni = plgpu.planar_snake(
          lin_idx,
          (m_iters, n_iters),
          config.grid_minor_dim,
          config.grid_tile_width,
      )

      wg = jax.lax.axis_index("wg")
      group_id = group_id_gmem[tid_m]
      start_within_block = start_within_block_gmem[tid_m]
      actual_size = actual_size_gmem[tid_m]
      block_start = block_start_gmem[tid_m]
      acc_slot = lax.rem(carry, jnp.int32(2))
      slice_m = pl.ds(block_start, block_m)
      slice_n = pl.ds(ni * block_n, block_n)
      slice_acc_m = pl.ds(acc_slot * block_m, block_m)

      is_lead_block = cluster_idx == 0

      @pl.when(actual_size > 0)
      def _body():

        @pl.when(wg == _COMPUTE_WG)
        def _():
          @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
          def _per_warp():
            warp_id = lax.axis_index("warp")

            @pl.when(warp_id == _TMA_WARP)
            def _memory():
              def _loop_body(ki, _):
                slice_k = pl.ds(ki * block_k, block_k)
                slot = lax.rem(ki, num_stages)

                @pl.when((ki >= num_stages) | (carry > 0))
                def _():
                  plgpu.barrier_wait(consumed_barrier.at[slot])

                plgpu.copy_gmem_to_smem(
                    x_gmem.at[slice_m, slice_k],
                    x_smem.at[slot],
                    xw_tma_barrier.at[slot],
                    partitioned_axis=0 if collective else None,
                    collective_axes="x" if collective else None,
                )
                plgpu.copy_gmem_to_smem(
                    w_gmem.at[group_id, slice_n, slice_k],
                    w_smem.at[slot],
                    xw_tma_barrier.at[slot],
                    partitioned_axis=0 if collective else None,
                    collective_axes="x" if collective else None,
                )

              lax.fori_loop(0, num_k_iters, _loop_body, None)

            @pl.when((warp_id == _MMA_WARP) & (carry > 1))
            def _wait_store():
              with jax.named_scope("wait for store"):
                plgpu.barrier_wait(store_done_barrier.at[acc_slot])

            @pl.when((warp_id == _MMA_WARP) & is_lead_block)
            def _mma():
              def _loop_body(ki, _):
                slot = lax.rem(ki, num_stages)
                with jax.named_scope("wait for xw"):
                  plgpu.barrier_wait(xw_tma_barrier.at[slot])
                with jax.named_scope("issuing mma"):
                  plgpu.tcgen05_mma(
                      acc_tmem.at[:, slice_acc_m],
                      w_smem.at[slot],
                      x_smem.at[slot].T,
                      consumed_barrier.at[slot],
                      accumulate=(ki > 0),
                      collective_axis="x" if collective else None,
                  )

                @pl.when(ki >= num_k_iters - 1)
                def _():
                  plgpu.tcgen05_commit_arrive(
                      mma_done_barrier.at[acc_slot],
                      collective_axis="x" if collective else None,
                  )

              lax.fori_loop(0, num_k_iters, _loop_body, None)

        @pl.when(wg == _STORE_WG)
        def _():
          plgpu.wait_smem_to_gmem(0, wait_read_only=True)
          plgpu.barrier_wait(mma_done_barrier.at[acc_slot])
          with jax.named_scope("tmem -> smem"):
            acc = plgpu.async_load_tmem(acc_tmem.at[:, slice_acc_m])
            plgpu.wait_load_tmem()
            if activation is not None:
              acc = activation(acc)
            acc = acc.astype(acc_smem.dtype)
            acc_smem.T[...] = plgpu.layout_cast(acc, _TCGEN05_TRANSPOSED)
            plgpu.commit_smem()

          with jax.named_scope("smem -> gmem"):
            # Write out the largest power of two rows first,
            # then the next largest, etc.
            # This allows us to coalesce writes as much as possible.
            offset = start_within_block
            size = 1 << (min(block_m, m).bit_length() - 1)
            while size > 0:

              @pl.when(actual_size & size != 0)
              def _():
                out_smem_slice = acc_smem.at[pl.ds(offset, size)]
                o_gref_slice = out_gmem.at[
                    pl.ds(block_start + offset, size),
                    pl.ds(ni * block_n + cluster_idx * tile_n, tile_n),
                ]
                plgpu.copy_smem_to_gmem(out_smem_slice, o_gref_slice)

              offset += actual_size & size
              size //= 2
            plgpu.wait_smem_to_gmem(0)
            plgpu.barrier_arrive(store_done_barrier.at[acc_slot])

      return carry + (actual_size > 0)

  def kernel_entry(*refs):

    def tiled_smem(shape, dtype, what=""):
      transforms = common.tile_swizzle_transforms(shape, dtype, what)
      return plgpu.SMEM(shape, dtype, transforms=transforms)

    x_smem = tiled_smem((num_stages, tile_m, block_k), x.dtype, "x")
    w_smem = tiled_smem((num_stages, tile_n, block_k), w.dtype, "w")
    acc_tmem = plgpu.TMEM(
        (tile_n, block_m * 2), dtype=jnp.float32, collective=collective
    )
    acc_smem = plgpu.SMEM(
        (block_m, tile_n),
        dtype=out_dtype,
        # workaround for ValueError: Dynamic slice base index (which is a
        # dynamic value) cannot be statically proven to be divisible by
        # the tiling (8)
        transforms=(
            plgpu.TilingTransform((1, 128 // jnp.dtype(out_dtype).itemsize)),
            plgpu.SwizzleTransform(128),
        ),
    )
    xw_tma_barrier = plgpu.Barrier(num_arrivals=2, num_barriers=num_stages)
    consumed_barrier = plgpu.Barrier(
        num_barriers=num_stages, orders_tensor_core=True
    )
    mma_done_barrier = plgpu.Barrier(num_barriers=2, orders_tensor_core=True)
    if collective:
      store_done_barrier = plgpu.ClusterBarrier(
          collective_axes=("x",),
          num_barriers=2,
          orders_tensor_core=True,
      )
    else:
      store_done_barrier = plgpu.Barrier(
          num_barriers=2, orders_tensor_core=True
      )
    pl.run_scoped(
        lambda *args: kernel(*refs, scoped=args),
        (x_smem, w_smem, acc_smem, acc_tmem),
        (
            xw_tma_barrier,
            consumed_barrier,
            mma_done_barrier,
            store_done_barrier,
        ),
        collective_axes="wg",
    )

  num_sms = backend.get_default_device().core_count
  profile = False
  f = plgpu.kernel(
      kernel_entry,
      out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
      num_threads=2,
      thread_name="wg",
      grid=(num_sms // 2,) if collective else (num_sms,),
      grid_names=("sm",),
      cluster=(1 + collective,),
      cluster_names=("x",),
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
          profile_space=30 if profile else 0,
          profile_dir="sponge" if profile else "",
      ),
  )
  return f(
      x,
      w,
      group_info.block,
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )



_MASK_WG = 0  # Warp group for masking RHS
_COMPUTE_WG_CONTRACTING = 1  # Warp group for TMA + MMA


def ragged_contracting_dim_dot_kernel_body_sm100(
    group_sizes_gmem,
    group_sizes_starts_gmem,
    lhs_gmem,
    rhs_gmem,
    o_gmem,
    *,
    scoped,
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
):
  """Pallas kernel body for non-quantized ragged contracting dim dot (SM100).

  Computes output[g] = lhs[group_start[g]:group_end[g], :].T @ rhs[group_start[g]:group_end[g], :]
  where lhs is (K, M) and rhs is (K, N), producing output (G, M, N).

  After transposing lhs to (M, K) and rhs to (N, K):
  - lhs_gmem is (M, K) in the kernel
  - rhs_gmem is (N, K) in the kernel
  - output[g] = rhs_T @ lhs_T.T = (N, K_g) @ (K_g, M) = (N, M), then transposed to (M, N)

  Architecture:
  - Two warp groups: MASK_WG applies ragged K mask, COMPUTE_WG does TMA+MMA
  - MASK_WG: loads RHS from SMEM (N, K), applies mask, stores to TMEM
  - COMPUTE_WG: TMA warp loads LHS/RHS, MMA warp uses masked RHS from TMEM
  """
  block_m = config.block_m
  block_k = config.block_k
  block_n = config.block_n
  num_stages = config.num_stages

  (lhs_smem, rhs_smem, acc_smem, rhs_masked_tmem, acc_tmem) = scoped[0]
  (
      load_barrier,
      consumed_barrier,
      mask_done_barrier,
      mma_done_barrier,
  ) = scoped[1]

  mi, ni, gi = map(jax.lax.axis_index, ("m", "n", "g"))
  group_start = group_sizes_starts_gmem[gi]
  group_end = group_start + group_sizes_gmem[gi]

  lb = jax.lax.div(group_start.astype(jnp.int32), block_k)
  ub = pl.cdiv(group_end.astype(jnp.int32), block_k)
  num_k_iters = ub - lb

  slice_m = pl.ds(mi * block_m, block_m)
  slice_n = pl.ds(ni * block_n, block_n)

  wg = jax.lax.axis_index("wg")

  # Guard: Only run computation if there are K iterations to process.
  # If num_k_iters == 0, all barriers would deadlock since signals happen
  # inside the loops which would execute 0 times.
  @pl.when(num_k_iters > 0)
  def _body():
    # Warp group 0: Masking warp group
    # Loads RHS from SMEM, applies ragged K mask, stores to TMEM.
    @pl.when(wg == _MASK_WG)
    def _mask_wg():
      def _mask_loop(ki, _):
        abs_ki = lb + ki
        k_start = abs_ki * block_k
        slot = lax.rem(ki, num_stages)

        # Wait for RHS to be loaded into SMEM.
        # Note: We don't need to wait on consumed_barrier here because:
        # 1. TMA warp waits on consumed_barrier before writing to SMEM
        # 2. TMA then signals load_barrier after writing
        # 3. By the time load_barrier fires, MMA has consumed previous data
        plgpu.barrier_wait(load_barrier.at[slot])

        # Load RHS tile from SMEM: (block_n, block_k) since RHS is transposed.
        rhs_tile = plgpu.load(rhs_smem.at[slot], ())

        # Compute mask for ragged K boundaries.
        # K is now axis 1 since RHS is (N, K) order.
        kx = plgpu.broadcasted_iota(
            jnp.int32, (block_n, block_k), 1, layout=_TCGEN05_TRANSPOSED
        )
        mask = (kx >= (group_start - k_start)) & (kx < (group_end - k_start))
        mask = mask.astype(rhs_tile.dtype)

        # Apply mask: rhs_masked is (block_n, block_k).
        rhs_masked = rhs_tile * mask

        # Store to TMEM: (block_n, block_k) matches TMEM slice shape.
        plgpu.async_store_tmem(
            rhs_masked_tmem.at[:, pl.ds(slot * block_k, block_k)], rhs_masked
        )
        plgpu.commit_tmem()
        plgpu.barrier_arrive(mask_done_barrier.at[slot])

      lax.fori_loop(0, num_k_iters, _mask_loop, None)

    # Warp group 1: Compute warp group (TMA + MMA)
    @pl.when(wg == _COMPUTE_WG_CONTRACTING)
    def _compute_wg():
      @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
      def _per_warp():
        warp_id = lax.axis_index("warp")

        @pl.when(warp_id == _TMA_WARP)
        def _tma():
          def _load_loop(ki, _):
            abs_ki = lb + ki
            slice_k = pl.ds(abs_ki * block_k, block_k)
            slot = lax.rem(ki, num_stages)

            @pl.when(ki >= num_stages)
            def _():
              plgpu.barrier_wait(consumed_barrier.at[slot])

            # Load LHS and RHS into SMEM.
            plgpu.copy_gmem_to_smem(
                lhs_gmem.at[slice_m, slice_k],
                lhs_smem.at[slot],
                load_barrier.at[slot],
            )
            # RHS is transposed (N, K), so slice as [slice_n, slice_k].
            plgpu.copy_gmem_to_smem(
                rhs_gmem.at[slice_n, slice_k],
                rhs_smem.at[slot],
                load_barrier.at[slot],
            )

          lax.fori_loop(0, num_k_iters, _load_loop, None)

        @pl.when(warp_id == _MMA_WARP)
        def _mma():
          def _mma_loop(ki, _):
            slot = lax.rem(ki, num_stages)

            # Wait for masked RHS in TMEM.
            plgpu.barrier_wait(mask_done_barrier.at[slot])

            # tcgen05_mma: acc += A @ B
            # A: (N, K) from TMEM (rhs already in N,K order)
            # B: (K, M) from lhs_smem transposed
            # Result: (N, K) @ (K, M) = (N, M)
            plgpu.tcgen05_mma(
                acc_tmem,
                rhs_masked_tmem.at[:, pl.ds(slot * block_k, block_k)],
                lhs_smem.at[slot].T,
                consumed_barrier.at[slot],
                accumulate=(ki > 0),
            )

            @pl.when(ki >= num_k_iters - 1)
            def _():
              plgpu.tcgen05_commit_arrive(mma_done_barrier)

          lax.fori_loop(0, num_k_iters, _mma_loop, None)

    # Epilogue: store accumulator to output (run by MASK_WG after MMA).
    @pl.when(wg == _MASK_WG)
    def _epilogue():
      plgpu.barrier_wait(mma_done_barrier)
      acc = plgpu.async_load_tmem(acc_tmem)
      plgpu.wait_load_tmem()
      if activation is not None:
        acc = activation(acc)
      acc = acc.astype(out_dtype)
      acc_smem.T[...] = plgpu.layout_cast(acc, _TCGEN05_TRANSPOSED)
      plgpu.commit_smem()
      plgpu.copy_smem_to_gmem(acc_smem, o_gmem.at[gi, slice_m, slice_n])
      plgpu.wait_smem_to_gmem(0, wait_read_only=True)


@jaxtyping.jaxtyped
def ragged_contracting_dim_dot_kernel_sm100(
    lhs: Float[Array, "K M"],
    rhs: Float[Array, "K N"],
    group_sizes: Integer[Array, "G"],
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: base.ActivationFunction | None = None,
) -> Float[Array, "G M N"]:
  """Pallas kernel for ragged contracting dim dot with non-quantized inputs (SM100)."""

  if lhs.dtype != rhs.dtype:
    raise NotImplementedError(
        f"lhs and rhs must have the same dtype. Got {lhs.dtype=} and"
        f" {rhs.dtype=}"
    )

  if lhs.dtype not in (jnp.bfloat16, jnp.float16):
    raise NotImplementedError(
        f"Only bfloat16/float16 inputs are supported. Got {lhs.dtype=}"
    )

  k, m = lhs.shape
  _, n = rhs.shape
  g = group_sizes.shape[0]

  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  num_stages = min(config.num_stages, k // block_k)

  def kernel_entry(
      group_sizes_ref,
      group_sizes_starts_ref,
      lhs_ref,
      rhs_ref,
      o_ref,
  ):
    def tiled_smem(shape, dtype, what=""):
      transforms = common.tile_swizzle_transforms(shape, dtype, what)
      return plgpu.SMEM(shape, dtype, transforms=transforms)

    lhs_smem = tiled_smem((num_stages, block_m, block_k), lhs.dtype, "lhs")
    # RHS is transposed at entry, so SMEM is (N, K) order.
    rhs_smem = tiled_smem((num_stages, block_n, block_k), rhs.dtype, "rhs")
    # TMEM buffer for masked RHS tiles (used by mask warp group).
    # Shape: (block_n, num_stages * block_k) to hold all staged masked tiles.
    rhs_masked_tmem = plgpu.TMEM(
        (block_n, num_stages * block_k), dtype=rhs.dtype, packed=True
    )
    acc_tmem = plgpu.TMEM((block_n, block_m), dtype=jnp.float32)
    acc_smem = plgpu.SMEM(
        (block_m, block_n),
        dtype=out_dtype,
        transforms=(
            plgpu.TilingTransform((1, 128 // jnp.dtype(out_dtype).itemsize)),
            plgpu.SwizzleTransform(128),
        ),
    )

    # Barriers for synchronization between warp groups and warps.
    load_barrier = plgpu.Barrier(num_arrivals=2, num_barriers=num_stages)
    consumed_barrier = plgpu.Barrier(
        num_barriers=num_stages, orders_tensor_core=True
    )
    mask_done_barrier = plgpu.Barrier(
        num_barriers=num_stages, orders_tensor_core=True
    )
    mma_done_barrier = plgpu.Barrier(num_barriers=1, orders_tensor_core=True)

    pl.run_scoped(
        lambda *args: ragged_contracting_dim_dot_kernel_body_sm100(
            group_sizes_ref,
            group_sizes_starts_ref,
            lhs_ref,
            rhs_ref,
            o_ref,
            scoped=args,
            out_dtype=out_dtype,
            config=config,
            activation=activation,
        ),
        (lhs_smem, rhs_smem, acc_smem, rhs_masked_tmem, acc_tmem),
        (load_barrier, consumed_barrier, mask_done_barrier, mma_done_barrier),
        collective_axes="wg",
    )

  kernel = plgpu.kernel(
      kernel_entry,
      out_shape=jax.ShapeDtypeStruct((g, m, n), out_dtype),
      num_threads=2,
      thread_name="wg",
      grid=(pl.cdiv(m, block_m), pl.cdiv(n, block_n), g),
      grid_names=("m", "n", "g"),
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          unsafe_no_auto_barriers=True,
      ),
  )

  group_sizes_starts = jnp.cumulative_sum(group_sizes, include_initial=True)
  # Transpose lhs from (K, M) to (M, K) and rhs from (K, N) to (N, K) for tcgen05_mma.
  return kernel(group_sizes, group_sizes_starts[:-1], lhs.T, rhs.T)
