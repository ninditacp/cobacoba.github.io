from redis import Redis
import rq
from ..topics import TopicModelling



def get_coherence_graph(target):
    topic = TopicModelling(target)
    topic.get_coherence_graph()

def topic_task(target):
    queue = rq.Queue('sm-worker', connection=Redis.from_url('redis://'))
    job = queue.enqueue('scripts.tasks.topics_task.get_coherence_graph', target, job_timeout='30m')
    return queue, job