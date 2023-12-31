from src.federatedscope.vertical_fl.xgb_base.worker import wrap_client_for_train, \
    wrap_server_for_train, wrap_client_for_evaluation, \
    wrap_server_for_evaluation


def wrap_vertical_server(server, config):
    if config.vertical.algo == 'xgb':
        server = wrap_server_for_train(server)
        server = wrap_server_for_evaluation(server)

    return server


def wrap_vertical_client(client, config):
    if config.vertical.algo == 'xgb':
        client = wrap_client_for_train(client)
        client = wrap_client_for_evaluation(client)

    return client
