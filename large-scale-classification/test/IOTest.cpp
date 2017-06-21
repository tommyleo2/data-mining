#include <gtest/gtest.h>
#include <memory>
#include <fstream>
#include <tuple>
#include <cstdlib>

#include "framework/Tree.hpp"

using namespace GBDT;

TEST(IOTest, NodeTest) {
  std::fstream tmp_file("tmp_in", std::fstream::out|std::fstream::binary);
  tmp_file << NONE << " " << 1 << " " << 2 << " " << 10000 << " " << 10.0 << std::endl;
  tmp_file << 0 << " " << NONE << " " << NONE << " " << 0.17 << std::endl;
  tmp_file << 0 << " " << NONE << " " << NONE << " " << 1.02 << std::endl;
  tmp_file.close();

  tmp_file = std::fstream("tmp_in", std::fstream::in|std::fstream::binary);
  Tree::Node tmp(nullptr);
  tmp_file >> tmp;

  EXPECT_EQ(NONE, tmp.m_parent);
  EXPECT_EQ(1, tmp.m_left);
  EXPECT_EQ(2, tmp.m_right);
  EXPECT_EQ(10000, std::get<0>(tmp.m_sp));
  EXPECT_EQ(10.0, std::get<1>(tmp.m_sp));

  tmp_file >> tmp;

  EXPECT_EQ(0, tmp.m_parent);
  EXPECT_EQ(NONE, tmp.m_left);
  EXPECT_EQ(NONE, tmp.m_right);
  EXPECT_EQ(0.17, tmp.m_weight);

  tmp_file >> tmp;

  EXPECT_EQ(0, tmp.m_parent);
  EXPECT_EQ(NONE, tmp.m_left);
  EXPECT_EQ(NONE, tmp.m_right);
  EXPECT_EQ(1.02, tmp.m_weight);

  tmp_file.close();

  tmp_file = std::fstream("tmp_out", std::fstream::out|std::fstream::binary);

  tmp.m_parent = NONE;
  tmp.m_left = 1;
  tmp.m_right = 2;
  tmp.m_sp = std::make_tuple(10000, 10.0);
  tmp_file << tmp << std::endl;

  tmp.m_parent = 0;
  tmp.m_left = NONE;
  tmp.m_right = NONE;
  tmp.m_weight = 0.17;
  tmp_file << tmp << std::endl;

  tmp.m_parent = 0;
  tmp.m_left = NONE;
  tmp.m_right = NONE;
  tmp.m_weight = 1.02;
  tmp_file << tmp << std::endl;

  tmp_file.close();

  EXPECT_EQ(0, system("diff tmp_in tmp_out"));
  remove("tmp_in");
  remove("tmp_out");
}
