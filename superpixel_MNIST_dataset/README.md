# Superpixel MNIST dataset

This dataset has been used to check the performance of the network when taking into account the intensities of the pixels. An important verification is made by comparing the MNIST images pixels intensities and the network output images pixel intensities.

Two different reconstruction loss terms are being tested here:

rec_loss(jrway) = sum_i min_j (d_E(p_pos(i), p'_pos(j)))^2 + sum_i min_j (d_E(p_pos(j), p'_pos(i)))^2 + sum_i (I(i) - I'(k'))^2 + sum_i (I(k) - I'(i))^2

where d_E is the euclidean distance p and p' are input and output pixels respectively, _pos is just the position of the pixels, I and I' are input and output pixels intensities respectively, and k' = argmin_j (d_E(p_pos(j), p'_pos(i)))^2, k = argmin_j (d_E(p_pos(i), p'_pos(j)))^2

rec_loss(thirdaxis) = sum_i min_j (d_E(p(i), p'(j)))^2 +  sum_i min_j (d_E(p(j), p'(i)))^2
