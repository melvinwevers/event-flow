import numpy as np
from scipy import stats


def jsd(p, q, base=np.e):
    '''Pairwise Jensen-Shannon Divergence for two probability distributions  
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return stats.entropy(p, m, base=base)/2. +  stats.entropy(q, m, base=base)/2.

def load_model(model):
    print("[INFO] reading model...")
    model_ = model_path + '{}_100_model.pcl'.format(model)
    with open(model_, "rb") as fobj:
        mdl = pickle.load(fobj)
    theta = mdl["theta"]
    time = mdl["dates"]
    return theta, time


def make_event_flow(model, window, rolling, normalize=True):
    print('loading jumps')
    jumps = pd.read_pickle('../models/jumps/{}_100_model_jump.pkl'.format(model))
    jumps.drop(0, axis=1, inplace=True) #remove point without jumps
    height = int(np.ceil(len(events) / 4))
    fig, axs = plt.subplots(height,4, figsize=(10, 100), facecolor='w', edgecolor='k', sharey=True)
    #fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()
    count = 0
    for key, value in events.items():
        jump_event = jumps[jumps['dates'] == value].drop('dates', axis=1).melt(var_name = 'jump_size', value_name = 'entropy')
        jump_event = jump_event[1500-window:1500+window]
        
        if normalize:
            jump_event = jump_event.subtract(jump_event.mean())
        
        axs[count].plot(jump_event['jump_size'], jump_event['entropy'], label=key)
        axs[count].plot(jump_event['jump_size'], jump_event['entropy'].rolling(rolling).mean().shift(int(np.floor(-.5 * rolling))), label=key)
        axs[count].set_title(key +' : ' +value)
        count += 1
    fig.tight_layout()
    axs[0].set_ylabel('entropy')
    axs[6].set_ylabel('entropy')
    plt.savefig('figures/{}_event_flow.png'.format(model))
    
    
def inspect_one_event(model, event, window, normalize=True):
    fig, axes = plt.subplots(1,2, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    
    jump_event = load_jump_event(model, event, window)
    
    if normalize:
        jump_event = jump_event.subtract(jump_event.mean())

    
    sns.scatterplot(jump_event['jump_size'], jump_event['entropy'], color='g', ax=axes[0])
    positive_ = jump_event[jump_event['jump_size'] > 0]['entropy']
    negative_ = jump_event[jump_event['jump_size'] < 0]['entropy']
    all_ = jump_event['entropy']
    
    plt.title(event)
    print(stats.ks_2samp(positive_, negative_))
    print('Kurtosis positive: {}'.format(kurtosis(positive_)))
    print('Skewness positive: {}'.format(skew(positive_)))
    print('Kurtosis negative: {}'.format(kurtosis(negative_)))
    print('Skewness negative: {}'.format(skew(negative_)))
    #print('skewness all: {}'.format(skew(all_)))
    print('kurtosis all: {}'.format(kurtosis(all_)))

    sns.distplot(positive_, kde=True, bins=25, label='positive', rug=True, color='r', ax=axes[1])
    sns.distplot(negative_, kde=True, bins=25, label='negative', rug=True, color='b', ax=axes[1])
    sns.distplot(all_, kde=True, bins=25, label='all', rug=True, color='black', ax=axes[1])
    plt.title(event)
    plt.legend()
    
def load_jump_event(model, event, window):
    jumps = pd.read_pickle('../models/jumps/{}_100_model_jump.pkl'.format(model))
    jumps.drop(0, axis=1, inplace=True) #remove point without jumps
    
    jump_event = jumps[jumps['dates'] == event].drop('dates', axis=1).melt(var_name = 'jump_size', value_name = 'entropy')
    jump_event = jump_event[1500-window:1500+window]
    return jump_event
    
    
def compare_newspaper(model_a, model_b, event, window):
    jump_a = load_jump_event(model_a, event, window)
    jump_b = load_jump_event(model_b, event, window)
    
    jump_a = jump_a.subtract(jump_a.mean())
    jump_b = jump_b.subtract(jump_b.mean())
    
    print('entire distribution')
    print(stats.ks_2samp(jump_a['entropy'], jump_b['entropy']))
    print('positive slopes')
    print(stats.ks_2samp(jump_a[jump_a['jump_size'] > 0]['entropy'], jump_b[jump_b['jump_size'] > 0]['entropy']))
          
    print('negative slopes')
    print(stats.ks_2samp(jump_a[jump_a['jump_size'] < 0]['entropy'], jump_b[jump_b['jump_size'] < 0]['entropy']))
    
def check_model(model):
    theta, time = load_model(model)
    entropy_df = daily_entropy(theta, time)
    print('First date: {}'.format(entropy_df.iloc[0]['ds']))
    print('Laste date: {}'.format(entropy_df.iloc[-1]['ds']))
    make_entropy_plot(entropy_df, model)
    monthly_entropy(entropy_df, model)
    
def compare_entropy(model_a, model_b, event, window, base=np.e):
    jump_a = load_jump_event(model_a, event, window)
    jump_b = load_jump_event(model_b, event, window)
    
    #jump_a = jump_a.subtract(jump_a.mean())
    #jump_b = jump_b.subtract(jump_b.mean())
      
    #p = jump_a[jump_a['jump_size'] 0]['entropy']
    #q = jump_b[jump_b['jump_size'] < 0]['entropy']
       
    
    p = np.asarray(jump_a['entropy'], dtype=np.float)
    q = np.asarray(jump_b['entropy'], dtype=np.float)
    
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    
    return stats.entropy(p, m, base=base)/2. +  stats.entropy(q, m, base=base)/2.
    
    
    
def daily_entropy(theta, time):
    entropies = np.zeros(theta.shape[0])
    for i in range(1, theta.shape[0]):
        entropies[i] = jsd(theta[i], theta[i-1])
    return pd.DataFrame(list(zip(time, entropies)),
                          columns=['ds','y'])

def make_entropy_plot(df, model):
    y = df['y'].rolling(30).mean()
    plt.figure(figsize=(20,10))
    plt.plot(df['ds'], df['y'], label='Entropy')
    plt.plot(df['ds'], y, label='rolling mean w=30')
    plt.ylabel('Entropy')
    plt.xlabel('Dates')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.title('Relative entropy between days')
    plt.savefig(f'figures/{model}_entropy_plot.png')
    plt.close()
    
def monthly_entropy(df, model):
    df['month'] = df['ds'].dt.month
    error_month = df.groupby('month')['y'].apply(lambda x:bootstrap.ci(data=x, statfunction=scipy.mean))

    upper_x = [x[1] for x in error_month]
    lower_x = [x[0] for x in error_month]
    mean_x = df.groupby('month')['y'].mean()
    
    plt.plot(mean_x)
    plt.title('Average entropy between days per month')
    plt.errorbar(mean_x.index, mean_x, yerr=[upper_x - mean_x, mean_x - lower_x], linestyle='')
    plt.ylabel('Entropy')
    plt.xlabel('Dates')
    plt.tight_layout()
    plt.show()
    #plt.title('Relative entropy between days')
    plt.savefig(f'figures/{model}_monthly_entropy.png')
    plt.close()
    
    
def fix_event_name(name):
    '''
    Function to Fix abbreviated event names into names used in viz.
    '''

    name = name.replace('_', ' ')
    print(name)
    ## some manual replacements
    if name == 'Biafra':
        name = 'Nigerian Civil War'
    elif name == 'Retreat vietnam':
        name = 'Fall of Saigon'
    elif name == 'EC nl':
        name = 'Euro 1988'
    elif name == 'wc argentina':
        name = 'Worldcup Soccer Argentina'
    capitalized_name = []
    for word in name.split():
        if len(word) <= 2:
            if word == 'of':
                capitalized_name.append(word)
            else:
                capitalized_name.append(word.upper())
        else:
            capitalized_name.append(word.title())
    name = ' '.join(capitalized_name)
    return name  


def changeSundays(date):
    '''
    This function changes the dates for events that took place on a Sunday. 
    On Sunday, newspapers did not publish.
    The date is shifted to the Saturday.
    '''
    day_ = datetime.datetime.strptime(date, '%Y-%m-%d').weekday()
    if day_ == 6:
        new_date = datetime.datetime.strptime(date, '%Y-%m-%d').date() + datetime.timedelta(days=1)
        return str(new_date)
    else:
        return str(date)