def init_nll_loss(log_prob):
    def nll_loss(params, inputs):
        # negative log-likelihood
        return -log_prob(params, inputs).mean()

    return nll_loss
