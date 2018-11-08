from moscrapy.neo4j.NeoModels import Movie, Genre, Actor, Director, Resource
from py2neo import Graph, NodeMatcher, RelationshipMatcher
from py2neo.ogm import GraphObject
import numpy as np
import random
import copy
import tensorflow as tf
import math
import json



def create_graph(graph):
    movie_to_actor = {
        'X战警：天启': [
            '詹姆斯·麦卡沃伊',
            '迈克尔·法斯宾德',
            '詹妮弗·劳伦斯',
            '尼古拉斯·霍尔特'
        ],
        'X战警：启示录': [
            '詹姆斯·麦卡沃伊',
            '迈克尔·法斯宾德',
            '詹妮弗·劳伦斯',
            '尼古拉斯·霍尔特'
        ],
        'X战警：逆转未来': [
            '休·杰克曼',
            '詹姆斯·麦卡沃伊',
            '迈克尔·法斯宾德',
            '詹妮弗·劳伦斯'
        ],
        '冬天的骨头': [
            '詹妮弗·劳伦斯',
            '约翰·浩克斯',
            '凯文·布雷斯纳汉',
            '戴尔·迪奇'
        ],
        '一出好戏': [
            '黄渤',
            '舒淇',
            '王宝强',
            '张艺兴',
        ],
        '纽约，我爱你': [
            '娜塔莉·波特曼',
            '舒淇',
            '海登·克里斯滕森',
        ]
    }
    # movie_to_actor = {
    #     '天启': [
    #         '詹妮弗·劳伦斯'
    #     ],
    #     '启示录': [
    #         '詹妮弗·劳伦斯'
    #     ],
    #     '一出好戏': [
    #         '黄渤'
    #     ]
    # }
    tx = graph.begin()
    mobj = {}
    aobj = {}
    for m_name in movie_to_actor:
        a_name_list = movie_to_actor[m_name]
        if m_name not in mobj:
            movie = Movie()
            movie.name = m_name
            mobj[m_name] = movie
        for a_name in a_name_list:
            if a_name not in aobj:
                actor = Actor()
                actor.name = a_name
                aobj[a_name] = actor
            # mobj[m_name].hasActor.add(aobj[a_name])
            # print('%s add actor %s' % (m_name, a_name))
    for m in mobj:
        print('push')
        tx.push(mobj[m])
    for a in aobj:
        tx.push(aobj[a])
    tx.commit()

    tx = graph.begin()
    mobj = {}
    aobj = {}
    for m_name in movie_to_actor:
        a_name_list = movie_to_actor[m_name]
        if m_name not in mobj:
            movie = list(Movie.match(graph).where('_.name="%s"' % m_name))[0]
            movie.name = m_name
            mobj[m_name] = movie
        for a_name in a_name_list:
            if a_name not in aobj:
                actor = list(Actor.match(graph).where('_.name="%s"' % a_name))[0]
                actor.name = a_name
                aobj[a_name] = actor
            mobj[m_name].hasActor.add(aobj[a_name])
            print('%s add actor %s' % (m_name, a_name))
    for m in mobj:
        tx.push(mobj[m])
    for a in aobj:
        tx.push(aobj[a])
    tx.commit()

class NeoUtil():

    def __init__(self, graph):
        self.graph = graph
        self.matcher = NodeMatcher(graph)
        self.relation_matcher = RelationshipMatcher(graph)

    # match (node)-[]->(rnode) return (rnode)
    def get_related_to_nodes(self, start_node):
        relations = self.relation_matcher.match([start_node])
        end_nodes = []
        for relation in relations:
            end_nodes.append(relation.nodes[1])
        return end_nodes

    # match (node)<-[]-(rnode) return (rnode)
    def get_related_from_nodes(self, end_node):
        relations = self.relation_matcher.match([None, end_node])
        start_nodes = []
        for relation in relations:
            start_nodes.append(relation.nodes[0])
        return start_nodes

    # match (node)-[]-(rnoode) return (rnode)
    def get_related_nodes(self, node):
        nodes = self.get_related_to_nodes(node)
        nodes.extend(self.get_related_from_nodes(node))
        return nodes
    '''
    node1_re -> [1, 2, 3]
    node2_re -> [2, 3, 4]
    return:
        list1 = [1, 1, 1, 0]
        list2 = [0, 1, 1, 1]
        id_set = (1, 2, 3, 4)
    '''
    def get_related_list(self, node1, node2):
        node1_re = self.get_related_nodes(node1)
        node2_re = self.get_related_nodes(node2)
        id_set = set()
        # id_set.add(node1.identity)
        # id_set.add(node2.identity)
        node1_id_obj = {}
        node2_id_obj = {}
        for node in node1_re:
            id_set.add(node.identity)
            node1_id_obj[node.identity] = 1
        for node in node2_re:
            id_set.add(node.identity)
            node2_id_obj[node.identity] = 1
        list1 = []
        list2 = []
        for id in id_set:
            list1.append(int(id in node1_id_obj))
            list2.append(int(id in node2_id_obj))
        return [list1, list2, id_set]

    def calc_dist(self, node1, node2):
        [list1, list2, ids] = self.get_related_list(node1, node2)
        return np.linalg.norm(np.array(list1) - np.array(list2))

    '''
    获取所有的triple， entity_id, relation_name    
    '''
    def get_all_triples(self):
        entity_id_set = set()
        relation_name_set = set()
        triple_list = []
        relations = self.relation_matcher.match()
        for relation in relations:
            start_node_id = relation.start_node.identity
            end_node_id = relation.end_node.identity
            relation_name = type(relation).__name__
            entity_id_set.add(start_node_id)
            entity_id_set.add(end_node_id)
            relation_name_set.add(relation_name)
            triple_list.append([start_node_id, relation_name, end_node_id])
        return triple_list, list(entity_id_set), list(relation_name_set)

    def get_all_triple_ids(self):
        triple_list, entity_list, relatioin_list = self.get_all_triples()
        entity_id_obj = {}
        relation_id_obj = {}
        triple_id_list = []
        triple_idc_obj = {}
        for idx in range(len(entity_list)):
            entity_id_obj[entity_list[idx]] = idx
        for idx in range(len(relatioin_list)):
            relation_id_obj[relatioin_list[idx]] = idx
        for [head, relation, tail] in triple_list:
            triple_id_list.append([entity_id_obj[head], relation_id_obj[relation], entity_id_obj[tail]])
            triple_idc_obj['%dc%dc%d' % (entity_id_obj[head], relation_id_obj[relation], entity_id_obj[tail])] = 1
        return triple_id_list, entity_id_obj, relation_id_obj, triple_idc_obj


class TransE():

    def __init__(self, triples, entities, relations, margin=1, dim=100):
        self.S = triples
        self.entities = entities
        self.relations = relations
        self.E = {}
        self.L = {}
        self.margin = margin
        self.dim = dim
        self.loss = 0
        for l in relations:
            # self.L[l] = np.random.uniform(-6/(self.dim**2), 6/(self.dim**2), self.dim)
            self.L[l] = np.random.uniform(-6 / np.sqrt(self.dim), 6 / np.sqrt(self.dim), self.dim)
        for e in entities:
            # self.E[e] = np.random.uniform(-6/(self.dim**2), 6/(self.dim**2), self.dim)
            self.E[e] = np.random.uniform(-6 / np.sqrt(self.dim), 6 / np.sqrt(self.dim), self.dim)
        self.triples_obj = {}
        for triple in triples:
            key = str(triple[0]) + str(triple[1]) + str(triple[2])
            self.triples_obj[key] = 1

    def run(self, max_loop=100000, batch_size=100, learning_rate=0.01, threshold=0.01):
        batch_generator = self.next_train_batch_generator(min(batch_size, len(self.S)))
        for loop_idx in range(max_loop):
            sample_batch, train_batch_corrupted = batch_generator.__next__()
            self.update(sample_batch, train_batch_corrupted, learning_rate)

            if loop_idx % 10 == 0:
                print("第%d次循环m, loss=%s" % (loop_idx, self.loss))
            if self.loss < threshold and self.loss != 0:
                break
        print('stop, loss = %s' % self.loss)

    def next_train_batch_generator(self, batch_size=10):
        while True:
            for e in self.E:
                self.E[e] = self.E[e]
            sample_batch = random.sample(self.S, batch_size)
            train_batch_corrupted = []
            for correct_triple in sample_batch:
                corrupted_triple = self.sample_corrupted_triplet(correct_triple)
                train_batch_corrupted.append(corrupted_triple)
            yield sample_batch, train_batch_corrupted


    def update(self, correct_triples, corrupted_triples, learning_rate):
        self.loss = 0
        length = len(correct_triples)
        # new_E = copy.deepcopy(self.E)
        # new_L = copy.deepcopy(self.L)
        for idx in range(length):
            correct_triple = correct_triples[idx]
            corrupted_triple = corrupted_triples[idx]
            head_array = self.E[correct_triple[0]]
            tail_array = self.E[correct_triple[2]]
            relation_array = self.L[correct_triple[1]]
            corrupted_head_array = self.E[corrupted_triple[0]]
            corrupted_tail_array = self.E[corrupted_triple[2]]

            l = self.margin + self.dist_L2(head_array, relation_array, tail_array) - self.dist_L2(corrupted_head_array, relation_array, corrupted_tail_array)
            # print(self.dist_L2(head_array, relation_array, tail_array))
            if l > 0:
                self.loss += l
                grad_h = 2 * (head_array + relation_array - tail_array)
                grad_t = -grad_h
                grad_c_h = -2 * (corrupted_head_array + relation_array - corrupted_tail_array)
                grad_c_t = -grad_c_h
                grad_l = grad_h - grad_c_h

                self.E[correct_triple[0]] = head_array - learning_rate * grad_h
                self.E[correct_triple[2]] = tail_array - learning_rate * grad_t
                self.L[correct_triple[1]] = relation_array - learning_rate * grad_l
                self.E[corrupted_triple[0]] = corrupted_head_array - learning_rate * grad_c_h
                self.E[corrupted_triple[2]] = corrupted_tail_array - learning_rate * grad_c_t

        #         new_E[correct_triple[0]] = head_array - learning_rate * grad_h
        #         new_E[correct_triple[2]] = tail_array - learning_rate * grad_t
        #         new_L[correct_triple[1]] = relation_array - learning_rate * grad_l
        #         new_E[corrupted_triple[0]]=  corrupted_head_array - learning_rate * grad_c_h
        #         new_E[corrupted_triple[2]] = corrupted_tail_array - learning_rate * grad_c_t
        # self.E = new_E
        # self.L = new_L

    def dist_L2(self, h, l, t):
        s = h + l - t
        return (s*s).sum()

    def sample_corrupted_triplet(self, triple):
        corrupted_triple = copy.copy(triple)
        # corrupt head node
        if np.random.randint(2) == 0:
            corrupted_head = np.random.choice(self.entities)
            while str(corrupted_head) + str(triple[1]) + str(triple[2]) in self.triples_obj:
                corrupted_head = np.random.choice(self.entities)
            corrupted_triple[0] = corrupted_head
        # corrupt tail node
        else:
            corrupted_tail = np.random.choice(self.entities)
            while str(triple[0]) + str(triple[1]) + str(corrupted_tail) in self.triples_obj:
                corrupted_tail = np.random.choice(self.entities)
            corrupted_triple[2] = corrupted_tail
        return corrupted_triple

    def norm_array(self, array):
        var = np.linalg.norm(array)
        return array / var
        # return array

# def test_tf(epochs_num=50000, learning_rate=0.01, margin=1, batch_size=10, dim=10):
#     neoUtil = NeoUtil(graph)
#     triples, entities, relations = neoUtil.get_all_triples()
#     for t in triples:
#         t[1] = 99
#     transE = TransE(triples, entities, relations)
#     generator = transE.next_train_batch_generator()
#     # vocabulary_size = len(entities) + len(relations)
#     vocabulary_size = 100
#
#     embedding_size = dim
#
#     def l1_energy(batch):
#         return tf.reduce_sum(tf.abs(batch[:, 0, :] + batch[:, 1, :] - batch[:, 2, :]), 1)
#
#     embedding_table = tf.Variable(
#         tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
#
#     pos_triples_batch = tf.placeholder(tf.int32, shape=[batch_size, 3])
#     neg_triples_batch = tf.placeholder(tf.int32, shape=[batch_size, 3])
#
#     pos_embed_batch = tf.nn.embedding_lookup(embedding_table, pos_triples_batch)
#     neg_embed_batch = tf.nn.embedding_lookup(embedding_table, neg_triples_batch)
#
#     p_loss = l1_energy(pos_embed_batch)
#     n_loss = l1_energy(neg_embed_batch)
#
#     loss = tf.reduce_sum(
#         tf.nn.relu(
#             margin + p_loss - n_loss))
#     optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
#
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         for step in range(epochs_num):
#             # p_triple, n_triple = next_train_batch(batch_size)
#             p_triple, n_triple = generator.__next__()
#             feed_dict = {pos_triples_batch: p_triple, neg_triples_batch: n_triple}
#             loss_val, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
#             print("step %d, loss_val %f" % (step, loss_val))


def write_triples_to_json():
    graph = Graph(uri='bolt://localhost:7687', auth=('neo4j', '123123'))

    neoUtil = NeoUtil(graph)
    triples, entities, relations, triple_idc = neoUtil.get_all_triple_ids()
    print(triples)
    print(entities)
    print(relations)
    print(triple_idc)

    with open('./triples.json', 'w') as f:
        f.write(json.dumps(triples))
    with open('./entities.json', 'w') as f:
        f.write(json.dumps(entities))
    with open('./relations.json', 'w') as f:
        f.write(json.dumps(relations))
    with open('./triple_idc.json', 'w') as f:
        f.write(json.dumps(triple_idc))




if __name__ == '__main__':
    # graph = Graph(uri='bolt://localhost:7687', auth=('neo4j', '123123'))
    # matcher = NodeMatcher(graph)

    pass

    # neoUtil = NeoUtil(graph)
    # triples, entities, relations, triple_idc = neoUtil.get_all_triple_ids()
    # print(triples)
    # print(entities)
    # print(relations)
    # print(triple_idc)



    # create_graph()

    # neoUtil = NeoUtil(graph)
    # triples, entities, relations = neoUtil.get_all_triples()
    # transE = TransE(triples, entities, relations, margin=1, dim=100)
    # transE.run(batch_size=1000, learning_rate=0.01)

    # print(transE.E)
    # print(transE.L)

    # def print_dist(n1, n2):
    #     vector1 = transE.E[n1]
    #     vector2 = transE.E[n2]
    #     res = np.linalg.norm(vector1 - vector2)
    #     print(res)
    #
    # def print_cos_dist(n1, n2):
    #     vector1 = transE.E[n1]
    #     vector2 = transE.E[n2]
    #     res = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
    #     print(res)

    # tianqi = transE.E[0]
    # qishilu = transE.E[42]
    # hasactor = transE.L['HAS_ACTOR']
    # jennifer = transE.E[75]
    # yichuhaoxi = transE.E[45]
    # huangbo = transE.E[81]
    #
    # print(transE.dist_L2(qishilu, hasactor, jennifer))
    # print(transE.dist_L2(qishilu, hasactor, huangbo))
    #
    # print()
    # print_dist(0, 42)
    # print_dist(0, 43)
    # print_dist(0, 45)









