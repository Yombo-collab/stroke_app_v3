import numpy as np


def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence de parité démographique entre les groupes.
    Retourne un dict avec 'difference', 'group_rates', 'groups'.
    """
    groups = np.unique(sensitive_attribute)
    group_rates = {}

    for group in groups:
        mask = sensitive_attribute == group
        if mask.sum() == 0:
            group_rates[group] = 0.0
        else:
            group_rates[group] = float(np.mean(y_pred[mask]))

    rates = list(group_rates.values())
    difference = max(rates) - min(rates)

    return {
        "difference": difference,
        "group_rates": group_rates,
        "groups": groups.tolist(),
    }


def disparate_impact_ratio(
    y_true, y_pred, sensitive_attribute, unprivileged_value, privileged_value
):
    """
    Calcule le ratio d'impact disproportionné (DI).
    DI = P(Y=1 | unprivileged) / P(Y=1 | privileged)
    Un ratio < 0.8 indique un biais significatif (règle des 4/5).
    """
    mask_unpriv = sensitive_attribute == unprivileged_value
    mask_priv = sensitive_attribute == privileged_value

    rate_unpriv = float(np.mean(y_pred[mask_unpriv])) if mask_unpriv.sum() > 0 else 0.0
    rate_priv = float(np.mean(y_pred[mask_priv])) if mask_priv.sum() > 0 else 1.0

    if rate_priv == 0:
        ratio = float("inf")
    else:
        ratio = rate_unpriv / rate_priv

    return {
        "ratio": ratio,
        "rate_unprivileged": rate_unpriv,
        "rate_privileged": rate_priv,
        "unprivileged_value": unprivileged_value,
        "privileged_value": privileged_value,
    }


def equalized_odds_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence d'égalité des chances (TPR gap entre groupes).
    """
    groups = np.unique(sensitive_attribute)
    tpr_by_group = {}

    for group in groups:
        mask = sensitive_attribute == group
        y_t = np.array(y_true)[mask]
        y_p = np.array(y_pred)[mask]
        positives = y_t == 1
        if positives.sum() == 0:
            tpr_by_group[group] = 0.0
        else:
            tpr_by_group[group] = float(np.mean(y_p[positives]))

    rates = list(tpr_by_group.values())
    difference = max(rates) - min(rates) if rates else 0.0

    return {
        "difference": difference,
        "tpr_by_group": tpr_by_group,
        "groups": groups.tolist(),
    }
