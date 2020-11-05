import pandas as pd 
import re 
import os
import numpy as np
from scipy import stats
from termcolor import colored

max_feasible_need = 66

def compute_situation(filename):
    with open(f'data/{filename}.txt') as f:
        txt = f.read().split('\n')

    states = []
    for i in range(len(txt)):
        if i % 6 == 0:
            if i > 0:
                states.append(current)
            current = [txt[i]]
        else:
            current.extend(re.findall(r'([\d]+,[\d]+|[\d]+)', txt[i].replace(' ', '')))

    columns = [
        'State',
        'Electoral votes',
        '% Reporting',
        '% Biden',
        'Votes Biden',
        '% Trump',
        'Votes Trump'
    ]
    df = pd.DataFrame(data=states, columns=columns)
    
    df.set_index('State', inplace=True)
    for col in df.columns:
        df[col] = df[col].replace(',', '.', regex=True).astype(float)

    def needed_for_victory(df, candidate, other):
        vote_diff = df[f'% {other}'].astype(float) - df[f'% {candidate}'].astype(float)
        leverage = 100/(100-df['% Reporting'].astype(float))
        votes_needed = 50 + vote_diff * leverage
        votes_needed[votes_needed < 0] = 0
        return leverage, votes_needed

    leverage, df['% Biden needs'] = needed_for_victory(df, 'Biden', 'Trump')
    _, df['% Trump needs'] = needed_for_victory(df, 'Trump', 'Biden')

    df.sort_index(inplace=True)
    leverage.sort_index(inplace=True)

    return leverage, df

def compute_electorals():
    with open('data/electorals.txt') as f:
        txt = f.read().split('\n')
    
    txt = [tx for tx in txt if len(tx) > 1]

    rs = []
    for tx in txt:
        state, votes_str = tx.split('-')
        votes = int(re.findall(r'[\d]+', votes_str)[0])
        rs.append([state.strip(), votes])

    result = pd.DataFrame(columns=['State', 'Electoral Votes'], data=rs).set_index('State')

    return result

if __name__ == "__main__":
    _, before = compute_situation('04-11-10am')
    leverage, after = compute_situation('last')

    diff = pd.DataFrame(
        columns=before.columns,
        index=before.index,
        data=after.to_numpy().astype(np.float) - before.to_numpy().astype(np.float))

    leverage.index = diff.index

    current = after
    
    current['# New votes'] = diff['Votes Biden'] + diff['Votes Trump']
    current['# Votes'] = after['Votes Biden'] + after['Votes Trump']

    assert np.alltrue((current['% Trump needs'] + current['% Biden needs']).values >= 99.9)

    new_vote_leverage = (1 - .01*current['% Reporting'].astype(float)) / (current['# New votes'] / current['# Votes'].astype(float))

    current['% Trump (new votes)'] = 50 + diff['% Trump'] * new_vote_leverage
    current['% Biden (new votes)'] = 100 - current['% Trump (new votes)']

    assert 0 < current['% Trump (new votes)'].min()
    assert current['% Trump (new votes)'].max() < 100

    winner = [
        'Trump' if tr_get > tr_need else 'Biden'
        for tr_get, tr_need in zip(current['% Trump (new votes)'], current['% Trump needs'])
    ]
    leader = [
        'Trump' if tr > bd else 'Biden'
        for tr, bd in zip(current['% Trump'], current['% Biden'])
    ]
    
    current['Leader'] = leader
    current['Heading'] = winner

    alpha = .99

    trump_need = current['% Trump needs'] / 100
    trump_up = (current['% Trump (new votes)'] - 50) / 100
    z_mlt = (trump_up*(1-trump_up)/current['# New votes']).apply(np.sqrt) / (.25/current['# New votes']).apply(np.sqrt)
    z_req = (trump_need*(1-trump_need)/current['# New votes']).apply(np.sqrt) / (.25/current['# New votes']).apply(np.sqrt)
    z_ci = stats.norm.ppf((1 + alpha) / 2) * (z_mlt - z_req)
    z_ci.fillna(0, inplace=True)

    z_alpha = stats.norm.ppf((1+alpha)/2)
    
    current['P(Trump win)'] = stats.norm.cdf(z_mlt)

    # +- alpha=.95 is the confidence interval at

    current['P(Trump win)'].replace(np.inf, 0, inplace=True)
    current['P(Trump win)'].replace(-np.inf, 0, inplace=True)
    current['P(Trump win)'].fillna(0, inplace=True)

    current['z (new votes)'] = (current['% Trump (new votes)']- current['% Trump needs']) / (current['P(Trump win)'])
    # current['P(Trump wins)'] = current['z (new votes)'].apply(stats.norm.cdf)

    print(current[[
        'Electoral votes',
        '% Reporting',
        '% Trump',
        '% Biden',
        '% Trump needs',
        '% Trump (new votes)',
        '# New votes',
        'P(Trump win)',
        #'P(Trump wins)',
        'Leader',
        'Heading'
        ]])

    electorals = compute_electorals()

    redblue = pd.read_csv(r'data/redbluestates.tsv', delimiter='\t').set_index('State')

    def compute_trump_biden(color):
        if color == 'swing':
            return color
        elif color == 'red':
            return 'Trump'
        return 'Biden'
    electorals['Traditional vote'] = redblue['Vote'].apply(compute_trump_biden)

    election_result = pd.DataFrame(
        columns=['Current', 'Heading'],
        index=electorals.index,
        data=[
            current.loc[state, ['Leader','Heading']] if state in current.index else [electorals.loc[state, 'Traditional vote']]*2
            for state in electorals.index
        ]
    )

    election_result['Electoral Votes'] = electorals['Electoral Votes']
    election_result['Traditional'] = redblue['Vote']
    election_result['Winnable Biden'] = current['% Biden needs'] <= current['% Biden (new votes)'] + z_alpha
    election_result['Winnable Biden'].fillna(election_result['Traditional'] == 'blue', inplace=True)
    election_result['Winnable Trump'] = current['% Trump needs'] <= current['% Trump (new votes)'] + z_alpha
    election_result['Winnable Trump'].fillna(election_result['Traditional'] == 'red', inplace=True)

    for ix in election_result.index:
        if not (
            election_result.loc[ix, 'Winnable Biden'] or
            election_result.loc[ix, 'Winnable Trump']):
            print('Unwinnable state:')
            print(election_result.loc[ix])
            print(f'Biden {alpha*100:.3f} % confidence bound:')
            print((current['% Biden (new votes)'] + z_alpha).loc[ix])
            print(f'Trump {alpha*100:.3f} % confidence bound:')
            print((current['% Trump (new votes)'] + z_alpha).loc[ix])
            print(f'z value: {z_alpha.loc[ix]:.3f}')

    assert np.alltrue([
        x or y for x, y in zip(
            election_result['Winnable Trump'].values,
            election_result['Winnable Biden'].values)])

    print(election_result)

    te = election_result['Electoral Votes'].sum()

    ct = election_result[election_result['Current'] == 'Trump']['Electoral Votes'].sum()
    ht = election_result[election_result['Heading'] == 'Trump']['Electoral Votes'].sum()

    cb = election_result[election_result['Current'] == 'Biden']['Electoral Votes'].sum()
    hb = election_result[election_result['Heading'] == 'Biden']['Electoral Votes'].sum()

    assert cb + ct == te
    assert hb + ht == te

    tht = election_result[election_result['Winnable Trump']]['Electoral Votes'].sum()
    thb = election_result[election_result['Winnable Biden']]['Electoral Votes'].sum()

    #assert tht >= max(ct, ht)
    #assert thb >= max(cb, hb)

    if tht < te / 2:
        thwin = 'Biden'
    elif thb < te / 2:
        thwin = 'Trump'
    else:
        thwin = 'Uncertain'

    print(f'Current: T: {ct}, B: {cb}, Winner: {"Trump" if ct > cb else "Biden"}')
    print(f'Heading: T: {ht}, B: {hb}, Winner: {"Trump" if ht > hb else "Biden"}')
    print(f'Theoretical: T: {tht}, B: {thb}, Winner: {thwin}')

    key_states = [ix for ix in election_result.index
        if (election_result.loc[ix, ['Current']].values[0] == 'Trump' and election_result.loc[ix, ['Winnable Biden']].values[0]
        or election_result.loc[ix, ['Current']].values[0] == 'Biden' and election_result.loc[ix, ['Winnable Trump']].values[0])]
    
    print('Key states:')
    print(current.loc[key_states,[
        'Electoral votes',
        '% Reporting',
        '% Trump',
        '% Biden',
        '% Trump needs',
        '% Trump (new votes)',
        '# New votes',
        'P(Trump win)',
        #'P(Trump wins)',
        'Leader',
        'Heading'
        ]])
