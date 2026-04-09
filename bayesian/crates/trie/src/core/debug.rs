use crate::symbol::Symbol;

use super::XNode;

#[cfg(test)]
pub(crate) fn format_node_slot_dump(node: &XNode, label: &str, symbol: Symbol) -> String {
    let slot = symbol.to_slot();
    let symbol_name = format!("{:?}", symbol);
    format!(
        concat!(
            "{}.c_can_trunc[{}] = {:?}\n",
            "{}.c_final_token_length[{}] = {:?}\n",
            "{}.c_final_token_lexindex[{}] = {:?}\n",
            "{}.c_p[{}] = {:?}\n",
            "{}.c_p_old[{}] = {:?}\n",
            "{}.c_fp[{}] = {:?}\n",
            "{}.c_tp[{}] = {:?}\n",
            "{}.c_tp0[{}] = {:?}\n",
            "{}.c_final_token_prob[{}] = {:?}\n",
            "{}.c_a_tl[{}] = {:?}\n",
            "{}.c_cuml_l_old[{}] = {:?}\n",
            "{}.c_cuml_l_old_for_mtcdl[{}] = {:?}\n",
            "{}.c_z[{}] = {:?}\n",
            "{}.c_a_pred_changed[{}] = {:?}\n",
            "{}.c_a_tp_changed[{}] = {:?}"
        ),
        label, symbol_name, node.c_can_trunc[slot],
        label, symbol_name, node.c_final_token_length[slot],
        label, symbol_name, node.c_final_token_lexindex[slot],
        label, symbol_name, node.c_p[slot],
        label, symbol_name, node.c_p_old[slot],
        label, symbol_name, node.c_fp[slot],
        label, symbol_name, node.c_tp[slot],
        label, symbol_name, node.c_tp0[slot],
        label, symbol_name, node.c_final_token_prob[slot],
        label, symbol_name, node.c_a_tl[slot],
        label, symbol_name, node.c_cuml_l_old[slot],
        label, symbol_name, node.c_cuml_l_old_for_mtcdl[slot],
        label, symbol_name, node.c_z[slot],
        label, symbol_name, node.c_a_pred_changed[slot],
        label, symbol_name, node.c_a_tp_changed[slot],
    )
}
