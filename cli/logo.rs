use crossterm::style::{Color, Stylize};

use crate::tui::{Line, rgb};

/// Titania, the moon, as pixel art after `.github/assets/hero.png`: a night
/// side (`d`), lit edge (`g`), and day side (`c`).
const MOON: [&str; 12] = [
    "....dgcc....",
    "..ddggcccc..",
    ".dddggcccgc.",
    ".ddddggcccc.",
    "dgddddggcccc",
    "gdgdddgggccc",
    "dgddddggggcc",
    "ddddddddggcc",
    ".dddddddggg.",
    ".ddddgdddgg.",
    "..dddddddg..",
    "....dddd....",
];

/// Green of the moon's lit edge, and Titania's accent color.
pub fn green() -> Color {
    rgb(62, 201, 138)
}

fn pixel(c: u8) -> Option<Color> {
    match c {
        b'd' => Some(rgb(20, 78, 58)),
        b'g' => Some(green()),
        b'c' => Some(rgb(253, 248, 225)),
        _ => None,
    }
}

/// The moon, drawn with half blocks: each character cell holds two pixels,
/// one above the other, which makes the pixels about square.
pub fn moon() -> Vec<Line> {
    MOON.chunks(2)
        .map(|rows| {
            let (top, bottom) = (rows[0].as_bytes(), rows[1].as_bytes());
            top.iter()
                .zip(bottom)
                .map(|(&top, &bottom)| match (pixel(top), pixel(bottom)) {
                    (Some(top), Some(bottom)) => "▀".to_string().with(top).on(bottom),
                    (Some(top), None) => "▀".to_string().with(top),
                    (None, Some(bottom)) => "▄".to_string().with(bottom),
                    (None, None) => " ".to_string().stylize(),
                })
                .collect()
        })
        .collect()
}
