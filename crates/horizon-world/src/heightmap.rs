//! Packed column heights and the six Java 26.3 heightmap predicates.
//!
//! Block behavior supplies air/fluid facts and resolved block-tag membership.
//! Heightmaps do not infer those facts from collision shapes or block names.
use horizon_core::storage::{PackedStorage, StorageError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HeightmapType {
    WorldSurfaceWg,
    WorldSurface,
    OceanFloorWg,
    OceanFloor,
    MotionBlocking,
    MotionBlockingNoLeaves,
}

impl HeightmapType {
    pub const ALL: [Self; 6] = [
        Self::WorldSurfaceWg,
        Self::WorldSurface,
        Self::OceanFloorWg,
        Self::OceanFloor,
        Self::MotionBlocking,
        Self::MotionBlockingNoLeaves,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::WorldSurfaceWg => "WORLD_SURFACE_WG",
            Self::WorldSurface => "WORLD_SURFACE",
            Self::OceanFloorWg => "OCEAN_FLOOR_WG",
            Self::OceanFloor => "OCEAN_FLOOR",
            Self::MotionBlocking => "MOTION_BLOCKING",
            Self::MotionBlockingNoLeaves => "MOTION_BLOCKING_NO_LEAVES",
        }
    }

    pub fn send_to_client(self) -> bool {
        matches!(
            self,
            Self::WorldSurface | Self::MotionBlocking | Self::MotionBlockingNoLeaves
        )
    }

    pub fn keep_after_worldgen(self) -> bool {
        !matches!(self, Self::WorldSurfaceWg | Self::OceanFloorWg)
    }

    pub fn is_opaque(self, state: impl HeightmapState) -> bool {
        match self {
            Self::WorldSurfaceWg | Self::WorldSurface => !state.is_air(),
            Self::OceanFloorWg | Self::OceanFloor => state.blocks_motion_in_heightmap(),
            Self::MotionBlocking => state.blocks_motion_in_heightmap() || state.has_fluid(),
            Self::MotionBlockingNoLeaves => {
                state.blocks_motion_in_heightmap_no_leaves() || state.has_fluid()
            }
        }
    }
}

pub trait HeightmapState: Copy {
    fn is_air(self) -> bool;
    /// Exact AIR block identity, distinct from cave/void air.
    fn is_air_block(self) -> bool;
    fn has_fluid(self) -> bool;
    fn blocks_motion_in_heightmap(self) -> bool;
    fn blocks_motion_in_heightmap_no_leaves(self) -> bool;
}

pub trait HeightmapChunk {
    type State: HeightmapState;
    fn min_y(&self) -> i32;
    fn height(&self) -> i32;
    fn highest_section_y(&self) -> i32;
    fn block_state(&self, x: i32, y: i32, z: i32) -> Self::State;
    /// When the 16-block section starting at `section_y` holds one state in
    /// every cell, returns that state so heightmap scans can resolve or skip
    /// the whole section in a single check.
    fn uniform_section(&self, _section_y: i32) -> Option<Self::State> {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeightmapError {
    InvalidBounds,
    MismatchedChunk,
    Storage(StorageError),
}
impl From<StorageError> for HeightmapError {
    fn from(value: StorageError) -> Self {
        Self::Storage(value)
    }
}
impl std::fmt::Display for HeightmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for HeightmapError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Heightmap {
    kind: HeightmapType,
    min_y: i32,
    height: i32,
    data: PackedStorage,
}

impl Heightmap {
    /// The heap bytes of its packed heights.
    pub fn heap_bytes(&self) -> usize {
        self.data.heap_bytes()
    }
    /// Bounded to the vanilla codec's positive, aligned vertical ranges.
    pub fn new(kind: HeightmapType, min_y: i32, height: i32) -> Result<Self, HeightmapError> {
        if !(-2032..=2031).contains(&min_y)
            || !(16..=4064).contains(&height)
            || min_y % 16 != 0
            || height % 16 != 0
            || min_y + height > 2032
        {
            return Err(HeightmapError::InvalidBounds);
        }
        let bits = (32 - (height as u32).leading_zeros()) as u8;
        Ok(Self {
            kind,
            min_y,
            height,
            data: PackedStorage::new(bits, 256)?,
        })
    }

    pub fn kind(&self) -> HeightmapType {
        self.kind
    }
    pub fn raw_data(&self) -> &[u64] {
        self.data.words()
    }

    fn index(x: i32, z: i32) -> usize {
        x.wrapping_add(z.wrapping_mul(16)) as usize
    }

    /// Whether `update` at (`x`, `y`, `z`) returns at once, whatever the
    /// state: the block is two or more below the first available one. `x`
    /// and `z` are within 0..16. Most generation writes are (the surface
    /// rule turns every stone under y 8 to deepslate), and this skips the
    /// checks `update` does first.
    #[inline]
    pub fn below_update(&self, chunk: &impl HeightmapChunk, x: i32, y: i32, z: i32) -> bool {
        debug_assert!((0..16).contains(&x) && (0..16).contains(&z));
        let first = (self.data.read(Self::index(x, z)) as i32).wrapping_add(self.min_y);
        y <= first.wrapping_sub(2) && self.check_chunk(chunk).is_ok()
    }

    pub fn first_available(&self, x: i32, z: i32) -> Result<i32, HeightmapError> {
        Ok((self.data.get(Self::index(x, z))? as i32).wrapping_add(self.min_y))
    }

    /// Whether any column of the local square `x0..=x1` by `z0..=z1` (each
    /// within 0..16) has its first available y at or above `y`.
    pub fn any_first_available_at_least(
        &self,
        [x0, x1]: [i32; 2],
        [z0, z1]: [i32; 2],
        y: i32,
    ) -> bool {
        debug_assert!(0 <= x0 && x1 < 16 && 0 <= z0 && z1 < 16);
        (z0..=z1).any(|z| {
            (x0..=x1)
                .any(|x| (self.data.read(Self::index(x, z)) as i32).wrapping_add(self.min_y) >= y)
        })
    }

    pub fn highest_taken(&self, x: i32, z: i32) -> Result<i32, HeightmapError> {
        Ok(self.first_available(x, z)?.wrapping_sub(1))
    }

    fn set_height(&mut self, x: i32, z: i32, y: i32) -> Result<(), HeightmapError> {
        self.data
            .set(Self::index(x, z), y.wrapping_sub(self.min_y) as u32)?;
        Ok(())
    }

    fn check_chunk(&self, chunk: &impl HeightmapChunk) -> Result<(), HeightmapError> {
        if chunk.min_y() != self.min_y || chunk.height() != self.height {
            return Err(HeightmapError::MismatchedChunk);
        }
        Ok(())
    }

    /// The caller writes the new state to the chunk before updating its maps.
    pub fn update<C: HeightmapChunk>(
        &mut self,
        chunk: &C,
        x: i32,
        y: i32,
        z: i32,
        state: C::State,
    ) -> Result<bool, HeightmapError> {
        self.check_chunk(chunk)?;
        let first = self.first_available(x, z)?;
        if y <= first.wrapping_sub(2) {
            return Ok(false);
        }
        if self.kind.is_opaque(state) {
            if y >= first {
                self.set_height(x, z, y.wrapping_add(1))?;
                return Ok(true);
            }
        } else if y == first.wrapping_sub(1) {
            let mut scan_y = y.wrapping_sub(1);
            while scan_y >= self.min_y {
                if (scan_y.wrapping_sub(self.min_y) & 15) == 15 {
                    let section_y = scan_y.wrapping_sub(15);
                    if let Some(uniform) = chunk.uniform_section(section_y) {
                        if self.kind.is_opaque(uniform) {
                            self.set_height(x, z, scan_y.wrapping_add(1))?;
                            return Ok(true);
                        }
                        scan_y = section_y.wrapping_sub(1);
                        continue;
                    }
                }
                if self.kind.is_opaque(chunk.block_state(x, scan_y, z)) {
                    self.set_height(x, z, scan_y.wrapping_add(1))?;
                    return Ok(true);
                }
                scan_y -= 1;
            }
            self.set_height(x, z, self.min_y)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Matching lengths preserve all bits, including padding and unused slots.
    /// A length mismatch primes this registered map from the chunk instead.
    pub fn set_raw_data(
        &mut self,
        chunk: &impl HeightmapChunk,
        words: &[u64],
    ) -> Result<(), HeightmapError> {
        self.check_chunk(chunk)?;
        if words.len() == self.data.words().len() {
            self.data = PackedStorage::from_words(self.data.bits(), 256, words.to_vec())?;
            Ok(())
        } else {
            Self::prime(chunk, &mut [self])
        }
    }

    /// Preserves caller map order and scans X, then Z, then descending Y.
    /// An unfound column retains its previous value, including during re-priming.
    pub fn prime(
        chunk: &impl HeightmapChunk,
        maps: &mut [&mut Self],
    ) -> Result<(), HeightmapError> {
        if maps.is_empty() {
            return Ok(());
        }
        for map in maps.iter() {
            map.check_chunk(chunk)?;
        }
        let min_y = chunk.min_y();
        let upper = chunk.highest_section_y().wrapping_add(16);
        if maps.len() <= 64 && upper >= min_y && (upper.wrapping_sub(min_y) & 15) == 0 {
            let full_mask = if maps.len() == 64 {
                u64::MAX
            } else {
                (1u64 << maps.len()) - 1
            };
            let section_count = ((upper - min_y) >> 4) as usize;
            // Cache per-section uniformity once across all 256 columns.
            let mut uniforms = Vec::with_capacity(section_count);
            for s in 0..section_count {
                let sy = min_y + (s as i32) * 16;
                uniforms.push(chunk.uniform_section(sy));
            }
            for x in 0..16 {
                for z in 0..16 {
                    let col = Self::index(x, z);
                    let mut remaining = full_mask;
                    for s in (0..section_count).rev() {
                        let section_y = min_y + (s as i32) * 16;
                        if let Some(state) = uniforms[s] {
                            if !state.is_air_block() {
                                let top_height = section_y.wrapping_add(16);
                                let mut bits = remaining;
                                while bits != 0 {
                                    let idx = bits.trailing_zeros() as usize;
                                    bits &= bits - 1;
                                    if maps[idx].kind.is_opaque(state) {
                                        maps[idx].set_height(x, z, top_height)?;
                                        remaining &= !(1u64 << idx);
                                    }
                                }
                                if remaining == 0 {
                                    break;
                                }
                            }
                            continue;
                        }
                        let mut y = section_y.wrapping_add(15);
                        while y >= section_y {
                            let state = chunk.block_state(x, y, z);
                            if !state.is_air_block() {
                                let height = y.wrapping_add(1);
                                let mut bits = remaining;
                                while bits != 0 {
                                    let idx = bits.trailing_zeros() as usize;
                                    bits &= bits - 1;
                                    if maps[idx].kind.is_opaque(state) {
                                        maps[idx]
                                            .data
                                            .write(col, height.wrapping_sub(min_y) as u32);
                                        remaining &= !(1u64 << idx);
                                    }
                                }
                                if remaining == 0 {
                                    break;
                                }
                            }
                            y -= 1;
                        }
                        if remaining == 0 {
                            break;
                        }
                    }
                }
            }
            return Ok(());
        }
        let mut remaining = Vec::with_capacity(maps.len());
        for x in 0..16 {
            for z in 0..16 {
                remaining.clear();
                remaining.extend(0..maps.len());
                let mut y = upper.wrapping_sub(1);
                while y >= min_y {
                    let state = chunk.block_state(x, y, z);
                    if !state.is_air_block() {
                        let mut i = 0;
                        while i < remaining.len() {
                            let map = &mut maps[remaining[i]];
                            if map.kind.is_opaque(state) {
                                map.set_height(x, z, y.wrapping_add(1))?;
                                remaining.remove(i);
                            } else {
                                i += 1;
                            }
                        }
                        if remaining.is_empty() {
                            break;
                        }
                    }
                    y -= 1;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Cell {
        Air,
        Leaves,
        Water,
        Stone,
    }

    impl HeightmapState for Cell {
        fn is_air(self) -> bool {
            matches!(self, Self::Air)
        }
        fn is_air_block(self) -> bool {
            matches!(self, Self::Air)
        }
        fn has_fluid(self) -> bool {
            matches!(self, Self::Water)
        }
        fn blocks_motion_in_heightmap(self) -> bool {
            matches!(self, Self::Leaves | Self::Stone)
        }
        fn blocks_motion_in_heightmap_no_leaves(self) -> bool {
            matches!(self, Self::Stone)
        }
    }

    struct MockChunk {
        min_y: i32,
        sections: Vec<Option<Box<[Cell; 4096]>>>,
        uniform: Vec<Option<Cell>>,
        report_uniform: bool,
    }

    impl MockChunk {
        fn new(min_y: i32, section_count: usize, report_uniform: bool) -> Self {
            Self {
                min_y,
                sections: vec![None; section_count],
                uniform: vec![Some(Cell::Air); section_count],
                report_uniform,
            }
        }
        fn fill_section(&mut self, s: usize, cell: Cell) {
            self.sections[s] = None;
            self.uniform[s] = Some(cell);
        }
        fn set(&mut self, x: i32, y: i32, z: i32, cell: Cell) {
            let s = ((y - self.min_y) >> 4) as usize;
            let fill = self.uniform[s].unwrap_or(Cell::Air);
            let data = self.sections[s].get_or_insert_with(|| Box::new([fill; 4096]));
            let idx = (((y - self.min_y) & 15) as usize) << 8 | ((z as usize) << 4) | (x as usize);
            data[idx] = cell;
            self.uniform[s] = None;
        }
    }

    impl HeightmapChunk for MockChunk {
        type State = Cell;
        fn min_y(&self) -> i32 {
            self.min_y
        }
        fn height(&self) -> i32 {
            self.sections.len() as i32 * 16
        }
        fn highest_section_y(&self) -> i32 {
            let top = self
                .sections
                .iter()
                .zip(&self.uniform)
                .rposition(|(data, u)| data.is_some() || *u != Some(Cell::Air))
                .unwrap_or(0);
            self.min_y + top as i32 * 16
        }
        fn block_state(&self, x: i32, y: i32, z: i32) -> Cell {
            let s = ((y - self.min_y) >> 4) as usize;
            if let Some(data) = &self.sections[s] {
                let idx =
                    (((y - self.min_y) & 15) as usize) << 8 | ((z as usize) << 4) | (x as usize);
                data[idx]
            } else {
                self.uniform[s].unwrap_or(Cell::Air)
            }
        }
        fn uniform_section(&self, section_y: i32) -> Option<Cell> {
            if !self.report_uniform {
                return None;
            }
            let s = ((section_y - self.min_y) >> 4) as usize;
            self.uniform[s]
        }
    }

    fn baseline_prime(
        chunk: &impl HeightmapChunk,
        maps: &mut [&mut Heightmap],
    ) -> Result<(), HeightmapError> {
        if maps.is_empty() {
            return Ok(());
        }
        for map in maps.iter() {
            map.check_chunk(chunk)?;
        }
        let upper = chunk.highest_section_y().wrapping_add(16);
        let mut remaining = Vec::with_capacity(maps.len());
        for x in 0..16 {
            for z in 0..16 {
                remaining.clear();
                remaining.extend(0..maps.len());
                let mut y = upper.wrapping_sub(1);
                while y >= chunk.min_y() {
                    let state = chunk.block_state(x, y, z);
                    if !state.is_air_block() {
                        let mut i = 0;
                        while i < remaining.len() {
                            let map = &mut maps[remaining[i]];
                            if map.kind.is_opaque(state) {
                                map.set_height(x, z, y.wrapping_add(1))?;
                                remaining.remove(i);
                            } else {
                                i += 1;
                            }
                        }
                        if remaining.is_empty() {
                            break;
                        }
                    }
                    y -= 1;
                }
            }
        }
        Ok(())
    }

    fn realistic_chunk(report_uniform: bool) -> MockChunk {
        // Overworld -64..320 (24 sections): solid deepslate/stone below, ocean/air gap,
        // sparse leaves/water canopy high up (forcing highest_section_y = 144).
        let mut c = MockChunk::new(-64, 24, report_uniform);
        for s in 0..6 {
            c.fill_section(s, Cell::Stone);
        }
        // Section 6 (y=32..47): ocean water
        c.fill_section(6, Cell::Water);
        // Sections 7..12 (y=48..143): air
        // Section 13 (y=144..159): a few high canopy leaves at sparse columns
        for x in (0..16).step_by(4) {
            for z in (0..16).step_by(4) {
                c.set(x, 150, z, Cell::Leaves);
            }
        }
        c
    }

    #[test]
    fn prime_and_update_with_uniform_sections_match_baseline() {
        let fast_chunk = realistic_chunk(true);
        let base_chunk = realistic_chunk(false);
        let kinds = [
            HeightmapType::WorldSurface,
            HeightmapType::OceanFloor,
            HeightmapType::MotionBlocking,
            HeightmapType::MotionBlockingNoLeaves,
        ];
        let mut base_maps: Vec<_> = kinds
            .iter()
            .map(|&k| Heightmap::new(k, -64, 384).unwrap())
            .collect();
        let mut fast_maps: Vec<_> = kinds
            .iter()
            .map(|&k| Heightmap::new(k, -64, 384).unwrap())
            .collect();

        let mut base_refs: Vec<&mut Heightmap> = base_maps.iter_mut().collect();
        let mut fast_refs: Vec<&mut Heightmap> = fast_maps.iter_mut().collect();
        baseline_prime(&base_chunk, &mut base_refs).unwrap();
        Heightmap::prime(&fast_chunk, &mut fast_refs).unwrap();
        assert_eq!(fast_maps, base_maps);

        // Removing a high canopy leaf scans down across 6 uniform air sections to water/stone.
        let mut fast_edited = realistic_chunk(true);
        let mut base_edited = realistic_chunk(false);
        fast_edited.set(0, 150, 0, Cell::Air);
        base_edited.set(0, 150, 0, Cell::Air);
        for (fast_map, base_map) in fast_maps.iter_mut().zip(base_maps.iter_mut()) {
            assert_eq!(
                fast_map.update(&fast_edited, 0, 150, 0, Cell::Air).unwrap(),
                base_map.update(&base_edited, 0, 150, 0, Cell::Air).unwrap(),
            );
            assert_eq!(fast_map, base_map);
        }
    }

    #[test]
    #[ignore = "microbenchmark: cargo test --release -p horizon-world --lib heightmap::tests::benchmark_heightmap_prime -- --ignored --nocapture"]
    fn benchmark_heightmap_prime() {
        use std::{hint::black_box, time::Instant};

        const ROUNDS: u32 = 31;
        const ITERS: usize = 500;
        let chunk = realistic_chunk(true);
        let kinds = [
            HeightmapType::WorldSurface,
            HeightmapType::OceanFloor,
            HeightmapType::MotionBlocking,
            HeightmapType::MotionBlockingNoLeaves,
        ];
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..ROUNDS {
            for optimized in [round % 2 == 0, round % 2 != 0] {
                let mut maps: Vec<_> = kinds
                    .iter()
                    .map(|&k| Heightmap::new(k, -64, 384).unwrap())
                    .collect();
                let mut refs: Vec<&mut Heightmap> = maps.iter_mut().collect();
                let started = Instant::now();
                for _ in 0..ITERS {
                    if optimized {
                        Heightmap::prime(black_box(&chunk), &mut refs).unwrap();
                    } else {
                        baseline_prime(black_box(&chunk), &mut refs).unwrap();
                    }
                    black_box(refs[0].raw_data()[0]);
                }
                if round > 0 {
                    times[usize::from(optimized)]
                        .push(started.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
                }
            }
        }
        for v in &mut times {
            v.sort_by(f64::total_cmp);
        }
        let median = |v: &[f64]| v[v.len() / 2];
        println!(
            "Heightmap::prime (4 maps, Overworld): baseline {:7.3} us/op  optimized {:7.3} us/op  speedup {:6.2}x",
            median(&times[0]),
            median(&times[1]),
            median(&times[0]) / median(&times[1])
        );
    }
}
