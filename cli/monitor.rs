use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use crossterm::style::Stylize;
use titania_model::Monitor;
use unicode_width::UnicodeWidthStr;

use crate::{
    logo,
    tui::{self, Line, span},
};

/// Most lines of disassembly the panel shows.
const LISTING: usize = 7;

/// How often the instruction rate is measured.
const RATE_INTERVAL: Duration = Duration::from_millis(500);

/// A panel showing what the simulated GPU is running: the kernel, where one
/// of its warps is, and how fast instructions are executing.
pub struct Panel {
    monitor: Arc<Monitor>,
    /// When the instruction count was last measured, and what it was.
    measured: (Instant, u64),
    /// Instructions per second since the measurement before.
    rate: f64,
}

impl Panel {
    pub fn new(monitor: Arc<Monitor>) -> Self {
        let instructions = monitor.activity().instructions();
        Self {
            monitor,
            measured: (Instant::now(), instructions),
            rate: 0.0,
        }
    }

    /// Draws the panel `columns` wide and at most `rows` tall, leaving out
    /// the disassembly if it doesn't fit, and everything if nothing does.
    pub fn draw(&mut self, columns: usize, rows: usize) -> Vec<Line> {
        let activity = self.monitor.activity();
        let instructions = activity.instructions();
        let (at, before) = self.measured;
        if at.elapsed() >= RATE_INTERVAL {
            self.rate = (instructions - before) as f64 / at.elapsed().as_secs_f64();
            self.measured = (Instant::now(), instructions);
        }

        let Some(kernel) = self.monitor.kernel() else {
            return Vec::new();
        };
        if rows < 4 {
            return Vec::new();
        }
        let sample = activity.sample();

        let title = " Titania GPU ";
        let mut lines = vec![vec![
            span("┌").dark_grey(),
            span(title).with(logo::green()),
            span(format!("{}┐", "─".repeat(columns.saturating_sub(2 + title.width())))).dark_grey(),
        ]];

        let grid = match kernel.grid {
            [width, 1] => width.to_string(),
            [width, height] => format!("{width}×{height}"),
        };
        let mut launch = format!("grid {grid} · block {}", kernel.block);
        if kernel.shared > 0 {
            launch.push_str(&format!(" · {} shared", bytes(kernel.shared)));
        }
        lines.push(tui::boxed(
            vec![span(&kernel.name).bold(), span("  "), span(launch).dark_grey()],
            columns,
        ));
        let position = format!("block {} · warp {}", sample.block, sample.warp);
        let stats = format!(
            "{} launches · {} instructions · {}/s",
            count(activity.launches() as f64),
            count(instructions as f64),
            count(self.rate),
        );
        let inner = columns.saturating_sub(4);
        lines.push(tui::boxed(tui::spread(vec![span(position)], vec![span(stats).dark_grey()], inner), columns));

        // The instructions around the one the sampled warp was at.
        let listing = LISTING.min(kernel.instructions.len()).min(rows.saturating_sub(5));
        if listing >= 3 {
            lines.push(tui::boxed(Line::new(), columns));
            let first = sample
                .pc
                .saturating_sub(listing / 2)
                .min(kernel.instructions.len() - listing);
            for (pc, text) in kernel.instructions.iter().enumerate().skip(first).take(listing) {
                let line = if pc == sample.pc {
                    vec![span("▶ ").with(logo::green()), span(format!("{pc:4}  {text}")).bold()]
                } else {
                    vec![span("  "), span(format!("{pc:4}  {text}")).dark_grey()]
                };
                lines.push(tui::boxed(line, columns));
            }
        }

        lines.push(vec![span(format!("└{}┘", "─".repeat(columns.saturating_sub(2)))).dark_grey()]);
        lines
    }
}

/// A count for humans: 950, 2.4K, 1.2M, 3.4G.
fn count(n: f64) -> String {
    const UNITS: [&str; 4] = ["K", "M", "G", "T"];
    if n < 1000.0 {
        return format!("{n:.0}");
    }
    let mut value = n / 1000.0;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.1}{}", UNITS[unit])
}

/// A size in bytes, for humans.
fn bytes(n: u32) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{} KiB", n / 1024)
    } else {
        format!("{n} B")
    }
}
