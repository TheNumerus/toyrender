use std::collections::VecDeque;

pub struct FrameReport {
    storage: Vec<(&'static str, FrameStatStorage, &'static str)>,
}

impl FrameReport {
    pub fn new() -> Self {
        Self { storage: Vec::new() }
    }

    pub fn log<T: FrameStat>(&mut self, value: T::Item) {
        self.storage.push((T::id(), FrameStatValue::into(value), T::display()));
    }
}

pub struct FrameStats {
    storage: VecDeque<FrameReport>,
    max_history: usize,
}

impl FrameStats {
    pub fn new(max_history: usize) -> Self {
        Self {
            storage: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    pub fn update(&mut self, frame_report: FrameReport) {
        if self.storage.len() >= self.max_history {
            self.storage.pop_front();
        }
        self.storage.push_back(frame_report);
    }

    pub fn compute(&self) -> Vec<(&'static str, StatStorage)> {
        if self.storage.is_empty() {
            return vec![];
        }

        let last_item = &self.storage[self.storage.len() - 1];

        let mut result = Vec::with_capacity(last_item.storage.len());

        for (id, val, display) in &last_item.storage {
            let stat = match val {
                FrameStatStorage::Float(_) => StatStorage::Float(self.compute_float(id)),
                FrameStatStorage::Int(_) => StatStorage::Int(self.compute_int(id)),
            };

            result.push((*display, stat));
        }

        result
    }

    fn compute_float(&self, id: &'static str) -> Stat<f32> {
        let mut history = 0.0;
        let mut latest = None;
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut avg = 0.0;

        for report in self.storage.iter().rev() {
            let mut value = None;
            for (old_id, old_val, _) in &report.storage {
                if id == *old_id {
                    value = Some(old_val);
                    break;
                }
            }

            match value {
                Some(FrameStatStorage::Float(value)) => {
                    avg += value;
                    history += 1.0;
                    min = min.min(*value);
                    max = max.max(*value);
                    if latest.is_none() {
                        latest = Some(*value);
                    }
                }
                _ => {
                    break;
                }
            }
        }

        avg /= history;

        Stat {
            latest: latest.unwrap(),
            min,
            max,
            avg,
        }
    }

    fn compute_int(&self, id: &'static str) -> Stat<u32> {
        let mut latest = None;
        let mut values = vec![];
        let mut min = u32::MAX;
        let mut max = u32::MIN;

        for report in self.storage.iter().rev() {
            let mut value = None;
            for (old_id, old_val, _) in &report.storage {
                if id == *old_id {
                    value = Some(old_val);
                    break;
                }
            }

            match value {
                Some(FrameStatStorage::Int(value)) => {
                    values.push(*value);
                    min = min.min(*value);
                    max = max.max(*value);
                    if latest.is_none() {
                        latest = Some(*value);
                    }
                }
                _ => {
                    break;
                }
            }
        }

        values.sort();
        let avg = values[values.len() / 2];

        Stat {
            latest: latest.unwrap(),
            min,
            max,
            avg,
        }
    }
}

pub enum StatStorage {
    Int(Stat<u32>),
    Float(Stat<f32>),
}

pub struct Stat<T> {
    pub latest: T,
    pub min: T,
    pub max: T,
    pub avg: T,
}

pub trait FrameStatValue {
    fn into(self) -> FrameStatStorage;
}
impl FrameStatValue for u32 {
    fn into(self) -> FrameStatStorage {
        FrameStatStorage::Int(self)
    }
}
impl FrameStatValue for f32 {
    fn into(self) -> FrameStatStorage {
        FrameStatStorage::Float(self)
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum FrameStatStorage {
    Float(f32),
    Int(u32),
}

pub trait FrameStat {
    type Item: FrameStatValue;
    fn id() -> &'static str;
    fn display() -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::{FrameReport, FrameStat, FrameStats, StatStorage};

    pub struct TestStat;
    impl FrameStat for TestStat {
        type Item = u32;
        fn id() -> &'static str {
            "test_stat"
        }

        fn display() -> &'static str {
            "Test Stat"
        }
    }

    pub struct FloatTestStat;
    impl FrameStat for FloatTestStat {
        type Item = f32;
        fn id() -> &'static str {
            "float_test_stat"
        }

        fn display() -> &'static str {
            "Float Test Stat"
        }
    }

    #[test]
    pub fn test_int_average() {
        let mut stats = FrameStats::new(3);

        let mut report = FrameReport::new();
        report.log::<TestStat>(64);
        stats.update(report);

        let mut report = FrameReport::new();
        report.log::<TestStat>(96);
        stats.update(report);

        let mut report = FrameReport::new();
        report.log::<TestStat>(110);
        stats.update(report);

        let result = stats.compute();
        assert_eq!(result[0].0, "Test Stat");
        if let StatStorage::Int(a) = &result[0].1 {
            assert_eq!(a.latest, 110);
            assert_eq!(a.max, 110);
            assert_eq!(a.min, 64);
            assert_eq!(a.avg, 96);
        } else {
            panic!("Wrong type of stat");
        }
    }

    #[test]
    pub fn test_float_average() {
        let mut stats = FrameStats::new(3);

        let mut report = FrameReport::new();
        report.log::<FloatTestStat>(2.0);
        stats.update(report);

        let mut report = FrameReport::new();
        report.log::<FloatTestStat>(3.0);
        stats.update(report);

        let mut report = FrameReport::new();
        report.log::<FloatTestStat>(5.0);
        stats.update(report);

        let result = stats.compute();
        assert_eq!(result[0].0, "Float Test Stat");
        if let StatStorage::Float(a) = &result[0].1 {
            assert_eq!(a.latest, 5.0);
            assert_eq!(a.max, 5.0);
            assert_eq!(a.min, 2.0);
            assert_eq!(a.avg, 10.0 / 3.0);
        } else {
            panic!("Wrong type of stat");
        }
    }

    #[test]
    pub fn test_discard_missing_history() {
        let mut stats = FrameStats::new(3);

        let mut report = FrameReport::new();
        report.log::<FloatTestStat>(2.0);
        stats.update(report);

        let report = FrameReport::new();
        stats.update(report);

        let mut report = FrameReport::new();
        report.log::<FloatTestStat>(5.0);
        stats.update(report);

        let result = stats.compute();
        assert_eq!(result[0].0, "Float Test Stat");
        if let StatStorage::Float(a) = &result[0].1 {
            assert_eq!(a.latest, 5.0);
            assert_eq!(a.max, 5.0);
            assert_eq!(a.min, 5.0);
            assert_eq!(a.avg, 5.0);
        } else {
            panic!("Wrong type of stat");
        }
    }
}
