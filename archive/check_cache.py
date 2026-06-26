"""
检查V7 embedding缓存是否基于错误的72k数据
"""
import joblib
import pandas as pd
from pathlib import Path

# 加载缓存
cache_path = Path("data/processed/v7_esm2_embeddings_72k.joblib")
if cache_path.exists():
    print(f"加载缓存: {cache_path}")
    cache = joblib.load(cache_path)
    cached_seqs = cache['sequences']
    embeddings = cache['embeddings']
    
    print(f"\n缓存信息:")
    print(f"  序列数: {len(cached_seqs)}")
    print(f"  嵌入形状: {embeddings.shape}")
    print(f"  前3条序列:")
    for i in range(3):
        print(f"    {i+1}: {cached_seqs[i][:50]}...")
    
    # 对比当前72k数据
    current_72k_path = Path("data/processed/v7_72k_quality_data.csv")
    if current_72k_path.exists():
        df_current = pd.read_csv(current_72k_path)
        current_seqs = df_current['sequence'].tolist()
        
        print(f"\n当前72k数据:")
        print(f"  序列数: {len(df_current)}")
        print(f"  突变数均值: {df_current['n_mutations'].mean():.1f}")
        
        # 检查是否匹配
        if len(cached_seqs) == len(current_seqs):
            match_first = cached_seqs[0] == current_seqs[0]
            match_last = cached_seqs[-1] == current_seqs[-1]
            match_middle = cached_seqs[len(cached_seqs)//2] == current_seqs[len(current_seqs)//2]
            
            print(f"\n匹配检查:")
            print(f"  第一条: {'OK' if match_first else 'FAIL'}")
            print(f"  中间条: {'OK' if match_middle else 'FAIL'}")
            print(f"  最后一条: {'OK' if match_last else 'FAIL'}")
            
            if match_first and match_last and match_middle:
                print(f"\n[OK] 缓存与当前72k数据匹配!")
                print(f"   但需要确认当前72k是新版本还是旧版本")
                
                # 检查突变数
                print(f"\n当前72k突变数分布:")
                print(f"  均值: {df_current['n_mutations'].mean():.1f}")
                print(f"  >=3突变: {(df_current['n_mutations'] >= 3).sum()} ({(df_current['n_mutations'] >= 3).sum()/len(df_current)*100:.1f}%)")
                
                if df_current['n_mutations'].mean() > 3.5:
                    print(f"\n[OK] 当前72k是**新版本**（突变数均值{df_current['n_mutations'].mean():.1f}）")
                    print(f"   缓存也是基于新数据生成的，不需要重新生成")
                else:
                    print(f"\n[WARN] 当前72k是**旧版本**（突变数均值{df_current['n_mutations'].mean():.1f}）")
                    print(f"   需要删除缓存并重新生成")
            else:
                print(f"\n[FAIL] 缓存与当前72k数据不匹配!")
                print(f"   需要删除缓存并重新生成")
        else:
            print(f"\n❌ 序列数不匹配! 缓存:{len(cached_seqs)}, 当前:{len(current_seqs)}")
            print(f"   需要删除缓存并重新生成")
    else:
        print(f"\n❌ 当前72k数据不存在: {current_72k_path}")
else:
    print(f"❌ 缓存不存在: {cache_path}")
    print(f"   V7训练时会自动生成缓存")
