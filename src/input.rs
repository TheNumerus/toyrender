use sdl2::keyboard::{KeyboardState, Scancode};
use std::collections::HashMap;
use std::hash::Hash;

pub struct InputMapper<K: Eq + Hash + Copy> {
    inner_state: HashMap<K, f32>,
    configuration: HashMap<Scancode, Vec<(K, f32)>>,
}

impl<K: Eq + Hash + Copy> InputMapper<K> {
    pub fn with_configuration(configuration: impl Into<HashMap<Scancode, Vec<(K, f32)>>>) -> Self {
        let configuration = configuration.into();

        let mut inner_state = HashMap::new();

        for value in configuration.values() {
            for (name, _) in value {
                inner_state.insert(*name, 0.0);
            }
        }

        Self {
            configuration,
            inner_state,
        }
    }

    pub fn update(&mut self, keyboard_state: KeyboardState) {
        let mut new_inputs = self.inner_state.keys().map(|&k| (k, 0.0)).collect::<HashMap<_, _>>();

        for (key, rules) in &self.configuration {
            if keyboard_state.is_scancode_pressed(*key) {
                for (axis, value) in rules {
                    *new_inputs.entry(*axis).or_default() += *value;
                }
            }
        }

        self.inner_state = new_inputs;
    }

    pub fn get_value(&self, axis: K) -> f32 {
        *self.inner_state.get(&axis).unwrap_or(&0.0)
    }
}
