#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define NUM_CHANNELS 8

void extract_times(unsigned char *data, int file_size, int channel, double *times, int *num_times) {
    double time = 0.0;
    bool prev_bit = false;
    for (int i = 0; i < file_size; i += 2) {
        int byte_index = i + (channel / 8);
        int bit_index = channel % 8;
        int bit = (data[byte_index] >> bit_index) & 1;
        if (channel == 2 && bit == 1 && prev_bit == false) {
            times[*num_times] = time;
            (*num_times)++;
            prev_bit = bit;
        }
        if (channel == 2) {
            prev_bit = bit;
        }
        // Sample rate is 20 kHz
        time += 0.00005;
    }
}

int main(int argc, char *argv[]) {
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(".")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (strstr(ent->d_name, ".dat") != NULL) {
                FILE *fp = fopen(ent->d_name, "rb");
                char* session_name = strtok(ent->d_name, ".");
                // Determine the size of the file in bytes
                fseek(fp, 0L, SEEK_END);
                long int file_size = ftell(fp);
                rewind(fp);

                // Allocate memory for the data
                unsigned char *data = (unsigned char *) malloc(file_size);
                if (data == NULL) {
                    printf("Error: could not allocate memory\n");
                    return 1;
                }

                // Read the data from the file
                size_t bytes_read = fread(data, sizeof(unsigned char), file_size, fp);
                if (bytes_read != file_size) {
                    printf("Error: could not read entire file\n");
                    return 1;
                }

                // Extract the binary information from each channel
                double time = 0.0;
                bool prev_bit_transition = false;
                bool prev_bit_cue = false;
                bool prev_bit_reward = false;
                bool prev_bit_no_reward = false;
                double *transition_times = malloc(sizeof(double) * file_size);
                double *cue_times = malloc(sizeof(double) * file_size);
                double *reward_times = malloc(sizeof(double) * file_size);
                double *no_reward_times = malloc(sizeof(double) * file_size);
                int num_times_transition = 0;
                int num_times_cue = 0;
                int num_times_reward = 0;
                int num_times_no_reward = 0;
                for (int i = 0; i < file_size; i += 2) {
                    for (int j = 0; j < NUM_CHANNELS; j++) {
                        int byte_index = i + (j / 8);
                        int bit_index = j % 8;
                        int bit = (data[byte_index] >> bit_index) & 1;
                        if (j == 0 && bit == 1 && prev_bit_transition == false) {
                            transition_times[num_times_transition] = time;
                            num_times_transition++;
                        }
                        if (j == 1 && bit == 1 && prev_bit_cue == false) {
                            cue_times[num_times_cue] = time;
                            num_times_cue++;
                        }
                        if (j == 2 && bit == 1 && prev_bit_reward == false) {
                            reward_times[num_times_reward] = time;
                            num_times_reward++;
                        }
                        if (j == 3 && bit == 1 && prev_bit_no_reward == false) {
                            no_reward_times[num_times_no_reward] = time;
                            num_times_no_reward++;
                        }
                        prev_bit_transition = (j == 0) ? bit : prev_bit_transition;
                        prev_bit_cue = (j == 1) ? bit : prev_bit_cue;
                        prev_bit_reward = (j == 2) ? bit : prev_bit_reward;
                        prev_bit_no_reward = (j == 3) ? bit : prev_bit_no_reward;
                    }
                    // Sample rate is 20 kHz
                    time += 0.00005;
                }

                // Write the times to a CSV file
                FILE *csv_fp = fopen(strcat(session_name, ".csv"), "w");
                if (csv_fp == NULL) {
                    printf("Error: could not open file\n");
                    return 1;
                }
                bool transition_print = false;
                fprintf(csv_fp, "Transition Times,Cue Times, No Reward Times, Reward Times\n");
                for (int i = 0; i < num_times_transition || i < num_times_cue || i < num_times_reward || i < num_times_no_reward;i++) {
                    if (i < num_times_transition) {
                        fprintf(csv_fp, "%.6f,", transition_times[i]);       
                        // printf("%s,%.6f\n", session_name, transition_times[i]);
                    } else {
                        fprintf(csv_fp, ",");
                    }
                    if (i < num_times_cue) {
                        fprintf(csv_fp, "%.6f,", cue_times[i]);
                    } else {
                        fprintf(csv_fp, ",");
                    }
                    if (i < num_times_no_reward) {
                        fprintf(csv_fp, "%.6f,", no_reward_times[i]);
                    } else {
                        fprintf(csv_fp, ",");
                    }
                    if (i < num_times_reward) {
                        fprintf(csv_fp, "%.6f", reward_times[i]);
                    }
                    fprintf(csv_fp, "\n");
                }
                fclose(csv_fp);

                // Free the memory and close the file
                free(data);
                fclose(fp);
            }
        }
        closedir(dir);
    } else {
        printf("Error: could not open directory\n");
        return 1;
    }
    
    return 0;
}


