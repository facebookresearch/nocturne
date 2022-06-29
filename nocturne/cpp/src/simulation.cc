// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "simulation.h"

#include "utils/sf_utils.h"

namespace nocturne {

void Simulation::Render() {
  if (render_window_ == nullptr) {
    constexpr int64_t kWinWidth = 1500;
    constexpr int64_t kWinHeight = 800;

    sf::ContextSettings settings;
    settings.antialiasingLevel = std::min(sf::RenderTexture::getMaximumAntialiasingLevel(), 4u);
    render_window_ = std::make_unique<sf::RenderWindow>(
        sf::VideoMode(kWinWidth, kWinHeight), "Nocturne", sf::Style::Default,
        settings);
    font_ = utils::LoadFont("Arial.ttf");
  }
  if (render_window_->isOpen()) {
    // parse events
    sf::Event event;
    while (render_window_->pollEvent(event)) {
      if (event.type == sf::Event::Closed) {
        render_window_->close();
        return;
      }
    }

    // clear window and set background color
    render_window_->clear(sf::Color(50, 50, 50));

    // draw scenario
    render_window_->draw(*scenario_);
  
    // draw frames per seconds on screen
    sf::Time elapsed = clock_.restart();
    float fps = 1.0f / elapsed.asSeconds();
    render_window_->setView(sf::View(sf::FloatRect(
        0, 0, render_window_->getSize().x, render_window_->getSize().y)));
    sf::Text text(std::to_string((int)fps) + " fps", font_, 20);
    text.setPosition(10, 5);
    text.setFillColor(sf::Color::White);
    render_window_->draw(text);

    // display everything that was drawn
    render_window_->display();
  } else {
    throw std::runtime_error(
        "tried to call the render method but the window is not open.");
  }
}

void Simulation::SaveScreenshot() {
  if (render_window_ != nullptr) {
    const std::string filename = "./screenshot.png";
    sf::Texture texture;
    texture.create(render_window_->getSize().x, render_window_->getSize().y);
    texture.update(*render_window_);
    texture.copyToImage().saveToFile(filename);
  }
}

}  // namespace nocturne
