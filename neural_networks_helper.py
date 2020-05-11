def conv_layer_output(W1, H1, D1, K, F, S, P):
    W2 = (W1 - F + 2  * P) // S + 1
    H2 = (H1 - F + 2 * P) // S + 1
    D2 = K
    return (W2, H2, D2)

def max_layer_output(W1, H1, D1, F, S):
    W2 = (W1 - F) // S + 1
    H2 = (H1 - F) // S + 1
    D2 = D1
    return (W2, H2, D2)
