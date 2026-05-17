import urllib.request, json, time, os, hashlib, re

os.makedirs('data', exist_ok=True)
base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
env = open('.env').read()
m = re.search(r'ENTREZ_EMAIL=(\S+)', env)
email = m.group(1) if m else 'test@test.com'

queries = [
    'cancer+gene+therapy',
    'BRCA1+breast+cancer',
    'EGFR+lung+cancer+treatment',
    'tumor+immunotherapy+checkpoint',
    'DNA+repair+pathway+cancer',
    'oncogene+mutation+drug',
    'chemotherapy+resistance+mechanism',
    'protein+kinase+inhibitor+cancer',
    'CRISPR+cancer+gene+editing',
    'metastasis+molecular+mechanism'
]

all_pmids = set()
for q in queries:
    url = base + f'esearch.fcgi?db=pubmed&term={q}&retmax=300&retmode=json&email={email}'
    try:
        r = urllib.request.urlopen(url, timeout=30)
        ids = json.loads(r.read())['esearchresult']['idlist']
        all_pmids.update(ids)
        print(f'Query [{q[:30]}]: {len(ids)} papers, total unique: {len(all_pmids)}')
        time.sleep(0.4)
    except Exception as e:
        print(f'Query error: {e}')

pmids = list(all_pmids)[:2000]
print(f'\nFetching abstracts for {len(pmids)} papers...')

existing = json.load(open('data/chunks.json')) if os.path.exists('data/chunks.json') else {'chunks':[]}
all_chunks = existing['chunks']
total_tokens = sum(len(c['text'].split()) for c in all_chunks)
print(f'Starting from {len(all_chunks)} existing chunks ({total_tokens} tokens)')

for i in range(0, len(pmids), 100):
    batch = pmids[i:i+100]
    ids_str = ','.join(batch)
    url2 = base + f'efetch.fcgi?db=pubmed&id={ids_str}&rettype=abstract&retmode=text&email={email}'
    try:
        r2 = urllib.request.urlopen(url2, timeout=60)
        text = r2.read().decode('utf-8', errors='ignore')
        words = text.split()
        total_tokens += len(words)
        for j in range(0, len(words), 300):
            chunk = ' '.join(words[j:j+300])
            if len(chunk) > 150:
                cid = hashlib.md5(f'full{i}{j}'.encode()).hexdigest()[:12]
                all_chunks.append({'id':cid,'text':chunk,'pmid':batch[0],'title':'PubMed Abstract','chunk_index':j})
        print(f'  Batch {i//100+1}/{len(pmids)//100+1} done — tokens: {total_tokens:,} chunks: {len(all_chunks)}')
    except Exception as e:
        print(f'  Batch error: {e}')
    time.sleep(0.4)
    if total_tokens >= 2100000:
        print('Reached 2M+ tokens target!')
        break

json.dump({'chunks': all_chunks}, open('data/chunks.json','w'))
print(f'\nFINAL: {len(all_chunks)} chunks, {total_tokens:,} tokens')
print('Run: python scripts/load_chromadb.py')
