# LOO-BCPD Pilot

LOO-BCPD is static full-burst inference. It uses frozen DRM plus WiSE image features, frozen class prototypes built from train labels, and no checkpoint update or OOD labels.

For target frame `i` and class `c`, frame responsibilities from other frames in the same burst produce:

\[
W_{-i,c}=\sum_{j\ne i} w_{j,c},\qquad
T_{-i,c}=\sum_{j\ne i}w_{j,c}q_j,
\]

\[
\delta_{-i,c}=\frac{T_{-i,c}-p_c(p_c^\top T_{-i,c})}{1+W_{-i,c}}.
\]

The tangent constraint keeps `p_c^T delta_{-i,c}=0`. The BCPD score is the cosine with the normalized displaced prototype; LOO-Linear removes only that normalization.

The pilot selects one BCPD strength on `IWildCamVal` from `0,0.25,0.5,1`, then evaluates the selected configuration without retuning. The diagnostics artifact includes STP-Mean, LOO-Linear, class-derangement, burst-shuffle, self-including, and unconstrained-mixing controls with paired sequence bootstrap intervals.

## BOPA Negative Control

BOPA's full centered visual-prototype basis is retained only as a negative theoretical control. A burst direction in the orthogonal complement has equal inner product with every centered visual class prototype, so symmetric attenuation yields a class-common affine transform of the TPA score. It cannot, by itself, create the required class-specific prototype ranking changes.
