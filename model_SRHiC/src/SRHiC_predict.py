#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import argparse
import h5py  # 1. h5py 임포트 추가
import gc

################################################## Added by HiHiC ######
parser = argparse.ArgumentParser(description='SRHiC prediction process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True, help='HiHiC directory')
required.add_argument('--model', type=str, default='SRHiC', metavar='SRHiC', required=True, help='model name')
required.add_argument('--ckpt_file', type=str, metavar='[2]', required=True, help='pretrained model (.meta)')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True, help='input batch size (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, help='GPU ID (default: 0)')
required.add_argument('--input_data', type=str, metavar='[5]', required=True, help='input npz file path')
required.add_argument('--output_data_dir', type=str, default='./output_enhanced', metavar='[6]', required=True, help='directory for saving output')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.makedirs(args.output_data_dir, exist_ok=True)
########################################################################

epoch_size = args.batch_size

def predict(test_file, meta_file, predict_save_dir):
    # 2. 텐서플로우 메모리 증식 방지 설정
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    
    with tf.Session(config=config) as sess:
        tf.local_variables_initializer().run()
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, meta_file.split('.meta')[0])
        graph = tf.get_default_graph()
        output_x = tf.get_collection('output_x')[0]
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        print("restoration is done...")

        test = np.load(test_file, mmap_mode='r')
        x_raw = test['data']
        num_samples = x_raw.shape[0]
        size = int(np.ceil(num_samples / epoch_size))
        
        prefix = os.path.splitext(os.path.basename(test_file))[0]      
        th_model = meta_file.split('/')[-1].split('_')[0]
        final_file = os.path.join(predict_save_dir, f'{prefix}_{args.model}_{th_model}ep.npz')
        h5_path = os.path.join(predict_save_dir, f"temp_{prefix}.h5")

        # 3. HDF5 캐시 설정 변경 (메모리 점유 방지)
        # rdcc_nbytes=0 으로 설정하여 하드디스크에 직접 쓰고 캐시하지 않도록 합니다.
        with h5py.File(h5_path, 'w', rdcc_nbytes=0) as f:
            dset = f.create_dataset('data', shape=(num_samples, 28, 28, 1), dtype='float32')
            
            print(f"Total batches: {size}")
            for z in range(size):
                start_idx = z * epoch_size
                end_idx = min((z + 1) * epoch_size, num_samples)
                
                # 딱 필요한 만큼만 읽어서 변환
                batch_x = x_raw[start_idx:end_idx].astype(np.float32)
                batch_x = batch_x[:, :, :40] 
                batch_x = np.reshape(batch_x, [-1, 40, 40, 1])
                
                out = sess.run(output_x, feed_dict={input_x: batch_x})
                
                # 디스크에 기록
                dset[start_idx:end_idx] = out.reshape([-1, 28, 28, 1])
                
                # 4. [중요] 주기적인 강제 메모리 비우기
                if z % 500 == 0:
                    print(f"Batch {z}/{size} saved. Force clearing memory...")
                    del out, batch_x  # 변수 명시적 삭제
                    gc.collect()      # 파이썬 가비지 컬렉터 강제 실행

        # 5. 마지막 저장 단계 (이 부분이 가장 위험하므로 조심스럽게 처리)
        print("Finalizing to npz...")
        # f['data'][:] 대신 루프를 돌며 조금씩 옮기거나, 메모리가 허용할 때만 로드
        with h5py.File(h5_path, 'r') as f:
            # np.savez_compressed는 내부적으로 전체 로드가 필요하므로, 
            # 만약 여기서 죽는다면 결과물을 그냥 .h5로 쓰시는 게 낫습니다.
            np.savez_compressed(final_file, data=f['data'][:], inds=test['inds'])
            
        if os.path.exists(h5_path):
            os.remove(h5_path)
        
        print(f"Prediction completed: {final_file}")

predict(args.input_data, args.ckpt_file, args.output_data_dir)