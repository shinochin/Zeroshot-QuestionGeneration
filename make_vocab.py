import json
import numpy as np
import pickle
import argparse

def main(args):
    dir_path = './data/wikidata/'
    file_names = ['wikidata_test_answerable_pos.json', 'wikidata_train_answerable_pos.json', 'wikidata_valid_answerable_pos.json']
    wordcount = {}
    entitycount = {}
    propertycount = {}
    for file in file_names:
        with open(dir_path + file, 'r') as f:
            for line in f:
                obj = json.loads(line)
                words = obj['question'].split()
                for w in words:
                    wordcount[w] = wordcount.get(w, 0) + 1
                words = obj['subject_type'].split()
                for w in words:
                    wordcount[w] = wordcount.get(w, 0) + 1
                words = obj['object_type'].split()
                for w in words:
                    wordcount[w] = wordcount.get(w, 0) + 1
                words = obj['predicate_phrase'].split()
                for w in words:
                    wordcount[w] = wordcount.get(w, 0) + 1

                s = obj['triple_ids'][0]
                r = obj['triple_ids'][1]
                o = obj['triple_ids'][2]
                entitycount[s] = entitycount.get(s, 0) + 1
                entitycount[o] = entitycount.get(o, 0) + 1
                propertycount[r] = propertycount.get(r, 0) + 1

    if args.word:
        glove_emb = {}
        with open('./data/wordembeddings/wikidata/glove.6B.100d.txt') as f:
            for line in f:
                line = line.split()
                glove_emb[line[0]] = [float(val) for val in line[1:]]

        wordvocab = ['<pad>', '<unk>', '<s>', '</s>']
        wordembedding = [[0.]*100, [0.]*100, [0.]*100, [0.]*100]
        for w in wordcount:
            if w in glove_emb:
                wordvocab.append(w)
                wordembedding.append(glove_emb[w])
            else:
                wordvocab.append(w)
                wordembedding.append([(np.random.rand() - 0.5)/100 for _ in range(100)])

        with open('./data/wikidata/word.vocab', 'w') as f:
            for w in wordvocab:
                f.write(w + '\n')

        np_we = np.empty((len(wordembedding), len(wordembedding[0])))
        np_we[:] = wordembedding
        np_we = np_we.astype(np.float32)
        with open('./data/wordembeddings/wikidata/glove100d.pkl', 'wb') as f:
            pickle.dump(np_we, f)

    if args.entity or args.relation:
        TRANSE = './checkpoints/transe/Wikidata/'
        rel_file = TRANSE + 'embeddings/dimension_100/transe/relation2vec.bin'
        ent_file = TRANSE + 'embeddings/dimension_100/transe/entity2vec.bin'
        rel_vocab_openke = TRANSE + 'knowledge-graphs/relation2id.txt'
        ent_vocab_openke = TRANSE + 'knowledge-graphs/entity2id.txt'
        rel_emb_openke = np.memmap(rel_file , dtype='float32', mode='r').reshape(-1, 100)
        ent_emb_openke = np.memmap(ent_file , dtype='float32', mode='r').reshape(-1, 100)
        rel2id = return_vocab_openke_wikidata(rel_vocab_openke)
        ent2id = return_vocab_openke_wikidata(ent_vocab_openke)

        propertyvocab = ['<unk_property>']
        propertyembedding = [[0.]*100]
        for p in propertycount:
            if p in rel2id:
                propertyvocab.append(p)
                propertyembedding.append(rel_emb_openke[rel2id[p]].tolist())
        np_pe = np.empty((len(propertyembedding), len(propertyembedding[0])))
        np_pe[:] = propertyembedding
        np_pe = np_pe.astype(np.float32)
        with open('./data/wikidata/property.vocab', 'w') as f:
            for p in propertyvocab:
                f.write(p + '\n')
        with open('./checkpoints/transe/Wikidata/rel_embeddings.pkl', 'wb') as f:
            pickle.dump(np_pe, f)

        entityvocab = ['<unk_entity>']
        entityembedding = [[0.]*100]
        for e in entitycount:
            if e in ent2id:
                entityvocab.append(e)
                entityembedding.append(ent_emb_openke[ent2id[e]].tolist())
        np_ee = np.empty((len(entityembedding), len(entityembedding[0])))
        np_ee[:] = entityembedding
        np_ee = np_ee.astype(np.float32)
        with open('./data/wikidata/entity.vocab', 'w') as f:
            for e in entityvocab:
                f.write(e + '\n')
        with open('./checkpoints/transe/Wikidata/ent_embedding.pkl', 'wb') as f:
            pickle.dump(np_ee, f)


def return_vocab_openke_wikidata(file):
    with open(file, 'r') as f:
        print(f.readline())
        ids = [l.strip().split('\t') for l in f]
        k, v = list(zip(*ids))
        v = list(map(int, v))
    return dict(zip(k, v))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make vocabulary')
    parser.add_argument('-word', '--word', action='store_true')
    parser.add_argument('-relation', '--relation', action='store_true')
    parser.add_argument('-entity', '--entity', action='store_true')
    args = parser.parse_args()
    main(args)
