from client import VectorEngineClient

client = VectorEngineClient()


def retrieve(prompt):
    if classify_prompt(prompt) == 'GENERAL':
        return retrieve_top_n_docs(prompt)
    else:
        return retrieve_active_alerts()


def classify_prompt(prompt):
    if prompt.startswith('GENERAL'):
        return 'GENERAL'
    else:
        return 'SYSTEM'


def retrieve_top_n_docs(query, top_k=1):
    return client.search(query=query, top_k=top_k)


def retrieve_active_alerts():
    return """
        CWS CPU Usage increased
        Entitlement heap increased overtime
        Broker memory leak
        5xx error has increased
        Page service CPU usage increased overtime
    """
