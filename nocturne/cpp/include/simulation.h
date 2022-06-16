// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "geometry/vector_2d.h"
#include "object.h"
#include "scenario.h"

namespace nocturne {

class Simulation {
 public:
  Simulation(
      const std::string& scenario_path,
      const std::unordered_map<std::string, std::variant<bool, int64_t, float>>&
          config)
      : scenario_path_(scenario_path),
        scenario_(std::make_unique<Scenario>(scenario_path, config)),
        config_(config) {}

  void Reset() { scenario_.reset(new Scenario(scenario_path_, config_)); }

  void Step(float dt) { scenario_->Step(dt); }

  void Render();

  Scenario* GetScenario() const { return scenario_.get(); }

  void SaveScreenshot();

 protected:
  void UpdateView(float padding = 100.0f) const;

  const std::string scenario_path_;
  std::unique_ptr<Scenario> scenario_ = nullptr;

  const std::unordered_map<std::string, std::variant<bool, int64_t, float>>
      config_;

  std::unique_ptr<sf::RenderWindow> render_window_ = nullptr;

  sf::Font font_;
  sf::Clock clock_;
};

}  // namespace nocturne
