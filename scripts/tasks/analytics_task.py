from redis import Redis
import rq
from ..topics import TopicModelling
from flask import redirect, url_for
import sys


def analytics(instagram, n_topic):
    topics = TopicModelling(instagram)
    scripts, div = topics.viz(int(n_topic))
    return scripts, div

def analytic_rq(instagram, n_topic):
    queue = rq.Queue('sm-worker', connection=Redis.from_url('redis://'))
    job = queue.enqueue(analytics, args=[instagram, n_topic], job_timeout='1h')
    return queue, job