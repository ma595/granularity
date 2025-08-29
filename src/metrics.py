def dummy_check_density(temperature, salinity):
    """Dummy density calculation"""
    return temperature * 0.1 + salinity * 0.05


def dummy_temperature_500m_30NS_metric(temperature):
    """Dummy temperature metric"""
    return temperature.mean()


def dummy_ACC_Drake_metric_2(velocity_u, ssh):
    """Dummy ACC Drake metric"""
    return velocity_u * ssh
