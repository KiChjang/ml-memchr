#[unsafe(no_mangle)]
pub extern "C" fn rs_memchr(p: *const u8, needle: u8, len: usize) -> isize {
    let repeated_ones: u64 = 0x0101_0101_0101_0101;
    let repeated_cs = repeated_ones * (needle as u64);
    let mask = repeated_ones << 7;

    let mut offset = 0;
    while offset + 32 <= len {
        let word0 = unsafe { (p.add(offset) as *const u64).read_unaligned() } ^ repeated_cs;
        let word1 = unsafe { (p.add(offset + 8) as *const u64).read_unaligned() } ^ repeated_cs;
        let word2 = unsafe { (p.add(offset + 16) as *const u64).read_unaligned() } ^ repeated_cs;
        let word3 = unsafe { (p.add(offset + 24) as *const u64).read_unaligned() } ^ repeated_cs;
        let r0 = (word0 - repeated_ones) & !word0;
        let r1 = (word1 - repeated_ones) & !word1;
        let r2 = (word2 - repeated_ones) & !word2;
        let r3 = (word3 - repeated_ones) & !word3;

        if (r0 | r1 | r2 | r3) & mask != 0 {
            if r0 & mask != 0 {
            } else if r1 & mask != 0 {
                offset += 8;
            } else if r2 & mask != 0 {
                offset += 16;
            } else {
                offset += 24;
            }
            break;
        }
        offset += 32;
    }

    while offset + 8 <= len {
        let word = unsafe { (p.add(offset) as *const u64).read_unaligned() } ^ repeated_cs;
        if ((word - repeated_ones) & !word) & mask != 0 {
            break;
        }
        offset += 8;
    }

    for i in offset..len {
        let byte = unsafe { p.add(i).read_unaligned() };
        if byte == needle {
            return i as isize
        }
    }

    -1
}

#[unsafe(no_mangle)]
pub extern "C" fn rs_safe_memchr(p: *const u8, needle: u8, len: usize) -> isize {
    let slice = unsafe { std::slice::from_raw_parts(p, len) };
    let repeated_ones = 0x0101_0101_0101_0101u64;
    let repeated_cs = repeated_ones * (needle as u64);
    let mask = repeated_ones << 7;
    let mut iter = slice.chunks_exact(32);
    let mut offset = 0isize;

    if let Some(idx) = iter.find_map(|chunk| {
        let word0 = u64::from_ne_bytes(chunk[0..8].try_into().unwrap()) ^ repeated_cs;
        let word1 = u64::from_ne_bytes(chunk[8..16].try_into().unwrap()) ^ repeated_cs;
        let word2 = u64::from_ne_bytes(chunk[16..24].try_into().unwrap()) ^ repeated_cs;
        let word3 = u64::from_ne_bytes(chunk[24..32].try_into().unwrap()) ^ repeated_cs;

        let r0 = (word0 - repeated_ones) & !word0;
        let r1 = (word1 - repeated_ones) & !word1;
        let r2 = (word2 - repeated_ones) & !word2;
        let r3 = (word3 - repeated_ones) & !word3;

        if r0 & mask != 0 {
            chunk[0..8]
                .into_iter()
                .enumerate()
                .find_map(|(idx, byte)| (*byte == needle).then_some(idx as isize + offset))
        } else if r1 & mask != 0 {
            chunk[8..16]
                .into_iter()
                .enumerate()
                .find_map(|(idx, byte)| (*byte == needle).then_some(idx as isize + offset + 8))
        } else if r2 & mask != 0 {
            chunk[16..24]
                .into_iter()
                .enumerate()
                .find_map(|(idx, byte)| (*byte == needle).then_some(idx as isize + offset + 16))
        } else if r3 & mask != 0 {
            chunk[24..32]
                .into_iter()
                .enumerate()
                .find_map(|(idx, byte)| (*byte == needle).then_some(idx as isize + offset + 24))
        } else {
            offset += 32;
            None
        }
    }) {
        return idx;
    }

    if iter.remainder().is_empty() {
        return -1;
    }

    let (chunks, remainder) = iter.remainder().as_chunks::<8>();

    if let Some(idx) = chunks.into_iter().find_map(|chunk| {
        let word = u64::from_ne_bytes(*chunk) ^ repeated_cs;

        if ((word - repeated_ones) & !word) & mask != 0 {
            chunk
                .into_iter()
                .enumerate()
                .find_map(|(idx, byte)| (*byte == needle).then_some(idx as isize + offset))
        } else {
            offset += 8;
            None
        }
    }) {
        return idx;
    }

    if remainder.is_empty() {
        return -1;
    }

    remainder
        .into_iter()
        .enumerate()
        .find_map(|(idx, byte)| (*byte == needle).then_some(idx as isize + offset))
        .unwrap_or(-1)
}
