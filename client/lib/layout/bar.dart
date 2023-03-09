import 'package:atra/views/login.dart';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

Widget build_app_bar(
    BuildContext context, List<Tab> tabs, List<Widget> children) {
  List<Widget> views = [
    for (Widget child in children)
      // Create a container for each child
      Container(
        // Set constraints for each container
        constraints: const BoxConstraints(
            minWidth: 5, maxWidth: 512, maxHeight: double.infinity),
        // Set padding for each container
        padding: const EdgeInsets.fromLTRB(25, 100, 25, 10),
        // Set alignment for each container
        alignment: Alignment.center,
        // Wrap each child in a SingleChildScrollView to enable scrolling
        child: child,
      )
  ];

  return DefaultTabController(
      length: children.length,
      child: Scaffold(
        body: NestedScrollView(
          headerSliverBuilder: (BuildContext context, bool innerBoxIsScrolled) {
            return <Widget>[
              SliverAppBar(
                expandedHeight: 100.0,
                floating: false,
                pinned: true,
                stretch: false,
                flexibleSpace: FlexibleSpaceBar(
                    centerTitle: true,
                    collapseMode: CollapseMode.parallax,
                    background: Image.network(
                      "https://images.pexels.com/photos/1044988/pexels-photo-1044988.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1",
                      fit: BoxFit.fitWidth,
                    )),
                actions: [
                  IconButton(
                      onPressed: () {
                        showDialog(
                            context: context,
                            builder: (BuildContext context) {
                              return const Dialog(child: LoginPage());
                            });
                      },
                      icon: const Icon(Icons.login)),
                  IconButton(
                      onPressed: () {
                        launchUrl(Uri.parse("https://a-ware.io/atra"));
                      },
                      icon: const Icon(Icons.help_outline))
                ],
              ),
              /*SliverPersistentHeader(
                delegate: _SliverAppBarDelegate(
                  TabBar(
                    indicatorSize: TabBarIndicatorSize.label,
                    tabs: tabs,
                  ),
                ),
                pinned: true,
                floating: false,
              ),*/
            ];
          },
          body: TabBarView(
            physics: const BouncingScrollPhysics(),
            children: views,
          ),
        ),
      ));
}

class _SliverAppBarDelegate extends SliverPersistentHeaderDelegate {
  // Declare the instance variable for the TabBar
  _SliverAppBarDelegate(this._tabBar);

  // Define the TabBar variable
  final TabBar _tabBar;

  // Define the minimum height of the header
  @override
  double get minExtent => _tabBar.preferredSize.height;
  // Define the maximum height of the header
  @override
  double get maxExtent => _tabBar.preferredSize.height;

  // Build the header
  @override
  Widget build(
      BuildContext context, double shrinkOffset, bool overlapsContent) {
    // Return the TabBar
    return _tabBar;
  }

  // When should the header rebuild?
  @override
  bool shouldRebuild(_SliverAppBarDelegate oldDelegate) {
    return false;
  }
}
