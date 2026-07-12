import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def calculate_eer(fpr, tpr, thresholds):
    """Calculates the exact operational Equal Error Rate (EER)."""
    fnr = 1 - tpr
    # Find the index where FPR and FNR intersect
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = min(fpr[idx], fnr[idx])
    return eer, thresholds[idx]

def generate_biometric_metrics(probe_embeddings, gallery_embeddings, probe_labels, gallery_labels):
    """
    Computes pair scores, plots the ROC curve, and calculates the EER.
    Assumes embeddings are already L2 normalized.
    """
    print("Calculating biometric similarity pairs...")
    # 1. Compute full pairwise cosine similarity matrix
    # Shape: [Num Probes, Num Gallery]
    similarity_matrix = np.dot(probe_embeddings, gallery_embeddings.T)
    
    # 2. Create the Ground Truth Mask (1 for Genuine Match, 0 for Imposter Match)
    # Shape: [Num Probes, Num Gallery]
    ground_truth_mask = (probe_labels[:, None] == gallery_labels[None, :]).astype(int)
    
    # 3. Flatten matrices to extract score lists for Scikit-Learn
    scores = similarity_matrix.flatten()
    y_true = ground_truth_mask.flatten()
    
    # 4. Compute ROC Curve points
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = calculate_eer(fpr, tpr, thresholds)
    
    # 5. Create Publication-Ready ROC Curve Plot
    plt.figure(figsize=(9, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess baseline')
    
    # Plot the EER intersection point
    plt.plot([eer], [1 - eer], marker='o', markersize=8, color="red", label=f'EER Sweet Spot = {eer*100:.2f}%')
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate (FAR)', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate (1 - FRR)', fontsize=12, labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) - Open-Set Gait Verification', fontsize=14, pad=15, weight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    output_path = "results/gait_verification_roc.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print("\n" + "="*40)
    print(f"BIOMETRIC VERIFICATION RESULTS:")
    print(f"--> ROC Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"--> Equal Error Rate (EER):     {eer*100:.2f}%")
    print(f"--> Optimal Threshold Score:    {eer_threshold:.4f}")
    print(f"--> Plot saved cleanly to:      {output_path}")
    print("="*40)

# --- MOCK TESTING HARNESS FOR IMMEDIATE EXECUTION ---
if __name__ == "__main__":
    # Generating 200 random mock test embeddings to ensure script runs out-of-the-box
    n_probes, n_gallery, dim = 100, 100, 256
    
    # Mock labels (Subjects 75 to 124)
    p_labels = np.random.randint(75, 125, size=n_probes)
    g_labels = np.random.randint(75, 125, size=n_gallery)
    
    # Generating vectors with structured correlation so it emulates your actual pipeline model
    p_embeds = np.random.randn(n_probes, dim)
    g_embeds = np.random.randn(n_gallery, dim)
    
    # Injecting identity separation correlation alignment
    for i in range(n_probes):
        match_idx = np.where(g_labels == p_labels[i])[0]
        if len(match_idx) > 0:
            g_embeds[match_idx[0]] = p_embeds[i] + np.random.randn(dim) * 0.4
            
    # L2 Normalization pass
    p_embeds /= np.linalg.norm(p_embeds, axis=1, keepdims=True)
    g_embeds /= np.linalg.norm(g_embeds, axis=1, keepdims=True)
    
    generate_biometric_metrics(p_embeds, g_embeds, p_labels, g_labels)