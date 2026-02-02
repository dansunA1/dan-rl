use std::collections::HashMap;

pub type F = f64;
/// Information dictionary for environment step
pub type Info = HashMap<String, f64>;

pub type InfoFrameMapMut<'a, const N: usize> = [&'a mut Info;N];
