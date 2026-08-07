use crate::app::UiMessage;
use crate::app::frame_stats::{FrameStats, StatStorage};
use crate::image::ImageResource;
use crate::mesh::MeshInstance;
use crate::scene::{Node, PointLight, Scene, Transform};
use imgui::Ui;
use nalgebra_glm::{Mat4, Vec3};
use std::collections::BTreeSet;
use std::slice::IterMut;

pub fn stats_tab(ui: &Ui, frame_stats: &FrameStats, delta: f32) {
    let stats = frame_stats.compute();

    ui.text(format!("FPS: {:>8.3} ms", 1.0 / delta));
    ui.text(format!("Frame time: {:>8.3} ms", delta * 1000.0));

    for (desc, value) in stats.iter() {
        match value {
            StatStorage::Int(i) => {
                ui.text(format!("{}: {}", desc, i.latest));
            }
            StatStorage::Float(f) => {
                ui.text(format!("{}: {:>8.3}", desc, f.avg));
            }
        }
    }
}

pub fn lights_tab(ui: &Ui, scene: &mut Scene) {
    if ui.button("Add light") {
        scene.nodes.push(
            Node::new()
                .add_component(Transform(Mat4::new_translation(&Vec3::from_element(0.0))))
                .add_component(PointLight {
                    color: Vec3::new(1.0, 0.1, 0.1),
                    intensity: 10.0,
                    radius: 0.1,
                }),
        )
    }

    for (index, node) in &mut scene.nodes.iter_mut().enumerate() {
        if let Some(pl) = node.get_component_mut::<PointLight>() {
            ui.color_edit3(format!("Color##{index}"), pl.color.as_mut());
            ui.input_float(format!("Intensity##{index}"), &mut pl.intensity).build();
            ui.input_float(format!("Radius##{index}"), &mut pl.radius).build();
        }

        if let Some(t) = node.get_component_mut::<Transform>() {
            let mut transform = t.0.data.0[3];

            if ui.input_float4(format!("Pos##{index}"), &mut transform).build() {
                t.0.data.0[3] = transform;
            }
        }

        ui.separator();
    }
}

pub fn textures_tab(ui: &Ui, textures: &[ImageResource]) {
    if let Some(tt) = ui.begin_table_with_flags("Textures", 4, imgui::TableFlags::SIZING_FIXED_FIT) {
        for tex in textures {
            ui.table_next_column();
            ui.text(&tex.name);
            ui.table_next_column();
            ui.text(format!("{}x{}", tex.data.width(), tex.data.height()));
            ui.table_next_column();
            ui.text(format!(
                "CPU Mem: {:.02}MB",
                (tex.data.as_bytes().len() as f32) / 1024.0 / 1024.0
            ));
            ui.table_next_column();
            ui.text(format!("{:?}", tex.data.color()));

            ui.table_next_row();
        }
        tt.end();
    }
}

pub fn scene_tab(ui: &Ui, meshes: IterMut<MeshInstance>, messages: &mut BTreeSet<UiMessage>) {
    if let Some(tt) = ui.begin_table_with_flags("Scene", 3, imgui::TableFlags::SIZING_FIXED_FIT) {
        for (index, mesh) in meshes.enumerate() {
            ui.table_next_column();
            if ui.checkbox(format!("##{}", index), &mut mesh.visible) {
                messages.insert(UiMessage::ReferenceRenderReset);
            }
            ui.table_next_column();
            ui.text(format!("'{}'", mesh.resource.name));
            ui.table_next_column();
            ui.text(format!(
                "'{:?}'",
                mesh.resource.culling_info.bb_max - mesh.resource.culling_info.bb_min
            ));

            ui.table_next_row();
        }
        tt.end();
    }
}
