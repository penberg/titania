# Titania

## Coding Style

* Never use `pub(crate)` in source code. A definition is either `pub` or not.
* Use one `use` per crate with nested paths (`use std::{path::Path, sync::Arc};`, not a `use` per item), grouped as `std`, external crates, then `crate`, separated by blank lines.
