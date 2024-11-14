import os
import time

import pysam
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def analyze_vcf(vcf_input):
    sv_calls = []
    f = open(vcf_input).readlines()
    for i, line in enumerate(f):
        if '#' in line:
            continue
        seq = line.strip().split('\t')
        chrname = seq[0]
        pos = seq[1]
        info = parse_info(seq[7])
        svtype, svlen, end = info['SVTYPE'], info['SVLEN'], info['END']
        sv_call = {
            # 'chrname': 'chr'+ chrname,
            'chrname': chrname,
            'pos': int(pos),
            'svtype': svtype,
            'sv_length': int(abs(svlen)),
            'sv_end': int(end)
        }
        sv_calls.append(sv_call)

    return sv_calls


def parse_info(seq):
    info = {'SVTYPE': '', 'SVLEN': 0, 'END': 0}
    for item in seq.split(';'):  # PRECISE;SVTYPE=INS;SVLEN=235;END=3176152;CIPOS=-8,8;CILEN=-2,2;RE=10;RNAMES=NULL
        key_value = item.split('=')
        if len(key_value) != 2:
            continue
        key, value = key_value
        if key in ['SVLEN', 'END']:
            try:
                info[key] = abs(int(value))
            except ValueError:
                pass
        elif key == 'SVTYPE':
            info[key] = value[:3]
    return info


def parse_cigar(cigar):
    cigar_operations = {
        0: 'M',
        1: 'I',
        2: 'D',
        3: 'N',
        4: 'S',
        5: 'H',
        6: 'P',
        7: '=',
        8: 'X',
        9: 'B'
    }

    parsed_cigar = []
    for operation, length  in cigar:
        if length == 0:
            continue
        if operation in cigar_operations:
            op = cigar_operations[operation]
            if parsed_cigar and parsed_cigar[-1][0] == op:
                parsed_cigar[-1] = (op, parsed_cigar[-1][1] + length)
            else:
                parsed_cigar.append((op, length))
        else:
            continue

    return parsed_cigar


def process_sv(sv_call, bam_file, base_output_dir, min_mapq):
    try:
        bam = pysam.AlignmentFile(bam_file, 'r')
        sv_type = sv_call['svtype']
        sv_start = sv_call['pos']
        sv_length = sv_call['sv_length']
        sv_end = sv_start + sv_length


        # sv_end = sv_call['sv_end']
        #
        # if sv_length == 0:
        #     sv_length = sv_end - sv_start
        #
        # else:
        #     sv_end = sv_start + sv_length



        sv_left = sv_start - sv_length
        sv_right = sv_end + sv_length
        chrname = sv_call['chrname']

        if sv_left <= 0:
            sv_left = sv_start
            sv_right = sv_end + 2 * sv_length
        elif sv_right <= 0:
            sv_right = sv_end
            sv_left = sv_start - 2 * sv_length
        else:
            sv_left = sv_start - sv_length
            sv_right = sv_end + sv_length

        all_final_list = []

        for read in bam.fetch(chrname, sv_left, sv_right):
            if read.mapq <= min_mapq:
                continue

            cigar = read.cigartuples
            parsed_cigar = parse_cigar(cigar)
            is_reverse = read.is_reverse
            ins_list, adjusted_ins_list = [], []
            del_list, adjusted_del_list = [], []
            normal_list, adjusted_normal_list = [], []
            reverse_normal_list = []
            dup_list = []

            shift_ins = 0
            shift_del = 0
            ref_start = read.reference_start
            ref_end = read.reference_end
            start = ref_start

            # ①Analyze the CIGAR String
            for c in parsed_cigar:
                if c[0] in ['M', 'D', '=', 'X']:
                    shift_ins += c[1]
                if c[0] == 'I':
                    ins_position = ref_start + shift_ins
                    ins_list.append([ins_position, c[1]])
                    if ref_end - start >= 0:
                        normal_list.append([start, ref_start + shift_ins - 1])
                    start = ref_start + shift_ins
                if c[0] in ['M', '=', 'X']:
                    shift_del += c[1]
                if c[0] == 'D':
                    del_start = ref_start + shift_del
                    del_end = ref_start + shift_del + c[1] - 1
                    if start < del_start:
                        normal_list.append([start, del_start - 1])
                    del_list.append([del_start, del_end])
                    shift_del += c[1]
                    start = ref_start + shift_del
            if ref_end - start >= 0:
                normal_list.append([start, ref_end])

            # ②Adjust ins_list to retain segments within the sv_left and sv_right regions.
            for i in range(len(ins_list) - 1, -1, -1):
                ins_start, length = ins_list[i]
                if sv_left <= ins_start <= sv_right:
                    adjusted_ins_list.append([ins_start, length, 'I'])
                    if sv_type == 'DUP' and length >= 50 and sv_start < ins_start < sv_end:
                        dup_list.append([ins_start, length, 'DUP'])
                        adjusted_ins_list.pop()
                    ins_list.pop(i)
            # print(sv_call)
            # print(f"adjusted_ins_list: {adjusted_ins_list}, dup_list: {dup_list}")

            # ③Adjust del_list to retain segments within the sv_left and sv_right regions.
            for item in del_list:
                del_start, del_end = item
                if del_end >= sv_left and del_start <= sv_right:
                    if del_start < sv_left:
                        del_start = sv_left
                    if del_end > sv_right:
                        del_end = sv_right
                    adjusted_del_list.append([del_start, del_end, 'D'])
            # print(f"DEL_list: {del_list}, adjusted_del_list: {adjusted_del_list}")

            # ④Adjust normal_list to retain segments within the sv_left and sv_right regions.
            for item in normal_list:
                norm_start, norm_end = item
                if norm_end >= sv_left and norm_start <= sv_right:
                    if norm_start < sv_left:
                        norm_start = sv_left
                    if norm_end > sv_right:
                        norm_end = sv_right
                    if is_reverse:
                        reverse_normal_list.append([norm_start, norm_end, 'M-'])
                    else:
                        adjusted_normal_list.append([norm_start, norm_end, 'M'])

            # ⑤Sorting of All Segments
            merged_list = adjusted_ins_list + adjusted_del_list + adjusted_normal_list + dup_list + reverse_normal_list
            merged_list.sort()
            if merged_list[0][0] > sv_left:
                merged_list.insert(0, [sv_left, merged_list[0][0], 'W'])
            if merged_list[-1][1] < sv_right:
                merged_list.append([merged_list[-1][1], sv_right, 'W'])

            new_list = []
            for item in merged_list:
                if item[2] == 'I':
                    new_list.append(['I', item[1]])
                elif item[2] == 'DUP':
                    new_list.append(['DUP', item[1]])
                else:
                    new_list.append([item[2], item[1] - item[0] + 1])
            all_final_list.append(new_list)
        # print(all_final_list)
        img_width = max(sum(length for _, length in sublist) for sublist in all_final_list)
        img_height = max(50, len(all_final_list) * 10)


        img = Image.new('RGB', (img_width, img_height), (255, 255, 255))

        y_offset = 0
        for final_list in all_final_list:
            x_offset = 0
            for op, length in final_list:
                color = (0, 0, 0)
                if op == 'I':
                    color = (255, 0, 0)
                elif op == 'D':
                    color = (0, 255, 0)
                elif op == 'M':
                    color = (0, 0, 255)
                elif op == 'M-':
                    color = (0, 255, 255)
                elif op == 'DUP':
                    color = (255, 255, 0)
                elif op == 'W':
                    color = (255, 255, 255)
                for x in range(length):
                    for y in range(10):
                        img.putpixel((x_offset + x, y_offset + y), color)
                x_offset += length
            y_offset += 10

        img = img.resize((224, 224), Image.Resampling.LANCZOS)

        output_dir = os.path.join(base_output_dir, sv_type)
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.join(output_dir, f"{chrname}_{sv_start}_{sv_type}.png")
        img.save(file_name)
        img.close()
        bam.close()
        print(f'Image saved as: {file_name}')

    except Exception as e:
        print(f"Error processing SV call {sv_call}: {e}")






if __name__ == '__main__':
    start_time = time.time()
    bam_file = ''
    vcf_file = ''
    base_output_dir = ''
    sv_types = ['INS', 'DEL', 'INV', 'DUP']

    for sv_type in sv_types:
        output_dir = os.path.join(base_output_dir, sv_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    min_mapq = 20

    sv_calls = analyze_vcf(vcf_file)

    batch_size = 100
    max_workers = 4

    for i in range(0, len(sv_calls), batch_size):
        batch_sv_calls = sv_calls[i:i + batch_size]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_sv, sv_call, bam_file, base_output_dir, min_mapq) for sv_call in
                       batch_sv_calls]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in future: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script Runtimes: {elapsed_time} s")
