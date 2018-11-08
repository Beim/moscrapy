import json
import random
import tensorflow as tf
import math

class TransE_tf():

    def __init__(self, config, next_batch):
        self.entity_size = config['entity_size']
        self.relation_size = config['relation_size']
        self.dim = config['dim']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.margin = config['margin']
        self.max_epoch = config['max_epoch']
        self.threshold = config['threshold']
        self.next_batch = next_batch

    def train(self):
        entity_embedding_table = tf.Variable(
            tf.truncated_normal([self.entity_size, self.dim], stddev=6.0 / math.sqrt(self.dim)))
        entity_embedding_table = entity_embedding_table / tf.norm(entity_embedding_table, axis=1, keepdims=True)
        relation_embedding_table = tf.Variable(
            tf.truncated_normal([self.relation_size, self.dim], stddev=6.0 / math.sqrt(self.dim)))
        relation_embedding_table = relation_embedding_table / tf.norm(relation_embedding_table, axis=1, keepdims=True)

        triples = tf.placeholder(tf.int32, shape=[self.batch_size, 3])
        corrupted_triples = tf.placeholder(tf.int32, shape=[self.batch_size, 3])

        head_embedding = tf.nn.embedding_lookup(entity_embedding_table, triples[:, 0])
        tail_embedding = tf.nn.embedding_lookup(entity_embedding_table, triples[:, 2])
        relation_embedding = tf.nn.embedding_lookup(relation_embedding_table, triples[:, 1])
        corrupted_head_embedding = tf.nn.embedding_lookup(entity_embedding_table, corrupted_triples[:, 0])
        corrupted_tail_embedding = tf.nn.embedding_lookup(entity_embedding_table, corrupted_triples[:, 2])

        loss = self.calc_loss(head_embedding, relation_embedding, tail_embedding, corrupted_head_embedding, corrupted_tail_embedding)
        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(init)
            for step in range(self.max_epoch):
                triple_ids, corrupted_triple_ids = self.next_batch(self.batch_size)
                with tf.device('/device:GPU:0'):
                    r, _ = sess.run([loss, optimizer], feed_dict={triples: triple_ids, corrupted_triples: corrupted_triple_ids})
                if step % 1000 == 0: print('step %d, loss = %s' % (step, r))
                if (r < self.threshold and r != 0):
                    print('step %d, loss = %s' % (step, r))
                    break

    def calc_loss(self, h, l, t, h_c, t_c):
        return tf.reduce_sum(
            tf.nn.relu(self.margin + self.L2_norm(h, l, t) - self.L2_norm(h_c, l, t_c))
        )

    def L2_norm(self, h, l, t):
        return tf.norm(h+l-t, axis=1)

def run(triples, entities, relations, triple_idc):
    def next_batch(batch_size):
        batch_size = min(len(triples), batch_size)
        triple_ids = random.sample(triples, batch_size)
        corrupted_triple_ids = []
        if random.randint(0, 1) == 0:
            for triple_id in triple_ids:
                ctriple = [random.randint(0, len(entities) - 1), triple_id[1], triple_id[2]]
                while '%dc%dc%d' % (ctriple[0], ctriple[1], ctriple[2]) in triple_idc:
                    ctriple = [random.randint(0, len(entities) - 1), triple_id[1], triple_id[2]]
                corrupted_triple_ids.append(ctriple)
        else:
            for triple_id in triple_ids:
                ctriple = [triple_id[0], triple_id[1], random.randint(0, len(entities) - 1)]
                while '%dc%dc%d' % (ctriple[0], ctriple[1], ctriple[2]) in triple_idc:
                    ctriple = [random.randint(0, len(entities) - 1), triple_id[1], triple_id[2]]
                corrupted_triple_ids.append(ctriple)
        return triple_ids, corrupted_triple_ids

    config = {
        'entity_size': len(entities),
        'relation_size': len(relations),
        'dim': 10,
        'batch_size': 10,
        'learning_rate': 0.01,
        'margin': 1,
        'max_epoch': 100000,
        'threshold': 0.01,
    }
    transE_tf = TransE_tf(config, next_batch)
    transE_tf.train()

def get_triples_from_json():
    with open('./triples.json', 'r') as f:
        triples = json.load(f)
    with open('./entities.json', 'r') as f:
        entities = json.load(f)
    with open('./relations.json', 'r') as f:
        relations = json.load(f)
    with open('./triple_idc.json', 'r') as f:
        triple_idc = json.load(f)
    return triples, entities, relations, triple_idc


if __name__ == '__main__':
    triples, entities, relations, triple_idc = get_triples_from_json()
    run(triples, entities, relations, triple_idc)