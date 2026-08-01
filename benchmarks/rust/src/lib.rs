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
