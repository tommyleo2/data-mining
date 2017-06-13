CC := g++
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin
DEPEND_DIR := .depends
CCFLAG := -std=c++11 -g -I./${INC_DIR}

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEPEND_FILES := $(SRC_FILES:$(SRC_DIR)/%.cpp=$(DEPEND_DIR)/%.d)
TARGET := $(BIN_DIR)/main

all: $(DEPEND_FILES) $(TARGET)

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(CC) $^ $(CCFLAG) -o $@

$(OBJ_FILES): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CC) $< $(CCFLAG) -c -o $@

$(DEPEND_FILES): $(DEPEND_DIR)/%.d : $(SRC_DIR)/%.cpp
	@mkdir -p $(DEPEND_DIR)
	@$(CC) $< $(CCFLAG) -MM -MT $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $<) > $@

-include $(DEPEND_FILES)

.PHONY: clean
.PHONY: run
.PHONY: all

run: all
	@mkdir -p data/predict_data
	@mkdir -p data/predict_result
	@mkdir -p data/training_data
	./${TARGET}

clean:
	rm bin/ build/ .depends/ -rf
